"""
SPH Conversion Tools

This module provides functions for converting gridded data to spherical harmonic
representation using pyshtools and optimized interpolation techniques.

Author: Gareth Roberts adding to what Sia Ghelichkhan had produced.
Date: 29 August 2025

Additions/tweaks by GR:
[1] Addition of lsq_inversion().
[2] Modifications to main(), inc. addition of the plotting scripts. 
[3] Adjustment of SPH normalizations (from 'ortho' to '4pi').
[4] Writing output of SPH coefficients for insertion into propmat code, which 
includes additional multiplicative sqrt(4pi) term. 

"""

import numpy as np
import pyshtools as pysh
from scipy.spatial import cKDTree
import pygmt
import matplotlib.pyplot as plt
import pandas as pd
import sys
import subprocess as sb

def main():
    """
    Main execution function for spherical harmonic conversion and output.

    This function:
    1. Converts scattered data to spherical harmonic representation
    2. Writes coefficients to output files in different formats
    """
    # Input parameters
    input_filename = "./for_sia_160725.dat"
    output_sphco = "./for_sia_160725.sph"
    output_fortran = "./for_sia_160725_fortran.dat"
    max_degree = 40
    header_description = "Converted from gridded data"
    
    # megan holdt's dynamic topographic grid (from the SI to her 2022 paper, degrees 1-40)
    megan = pygmt.load_dataarray('./degree_1_to_40.grd')
    megan_xyz = pygmt.grd2xyz(grid=megan, output_type='pandas')

    # Convert scattered data to spherical harmonics
    shtools_sh = convert_grid_to_sph(input_filename, max_degree)
    shtools_sh.info()
    sh_global_grid = shtools_sh.expand(grid='DH2')
    
    # Write coefficients in SPHCO format
    w_propmat_sph_pyshtools(output_sphco, header_description, shtools_sh)

    # Write coefficients in Fortran-compatible format
    write_fortran_sph_format(output_fortran, header_description, shtools_sh)
    
    # Perform LSQ with the scattered data to generate spherical harmonic coefficients
    cilm_b = lsq_inversion(megan_xyz, max_degree)
    cilm_b_grid = cilm_b.expand(grid='DH2')
    
    # reading output from propmat back into pysh
    cmd = " awk '{if (NR>4) print $2, $3, $4, $5}' < SPH_REF_dyntopography > SPH_REF_dyntopography_cut "
    output = sb.check_output(cmd, stderr=sb.STDOUT, shell=True)
    sys.stdout.write('{}'.format(output))

    # do the re-normalization to convert from sia's formatting to 4pi normalization, using / np.sqrt(4*np.pi)
    coeffs_global_propmat = pysh.SHCoeffs.from_file('SPH_REF_dyntopography_cut', normalization='4pi', csphase = 1, lmax=50, format='shtools')  / np.sqrt(4*np.pi) / 1e3
    propmat_grid_global = coeffs_global_propmat.expand(grid='DH2')

# plotting with shtools and gmt

# plot power spectrum 
#   power_global = cilm_b_grid.spectrum()
#     fig, ax = cilm_b.plot_spectrum(unit='per_lm', convention='power')
#     plt.show()

# plotting maps
    fig = pygmt.Figure()
    region="g"
    proj="N0/8i"
    frame=True

    with fig.subplot(nrows=3, ncols=2, figsize=("45c", "35c"), autolabel=True):
       pygmt.makecpt(cmap="polar", series=[-2, 2, 0.01], continuous=True, background=True)

       # a
       panel=[0,0] 
       fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
       fig.grdimage(grid=megan,region=region,projection=proj,frame=frame, panel=panel)
       fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)
       fig.plot(data='./for_sia_160725.dat',region=region,projection=proj,style='c0.01i', fill='darkgreen',panel=panel)

       # b       
       panel=[0,1] 
       fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
       fig.grdimage(grid=cilm_b_grid.to_xarray(),region=region,projection=proj,frame=frame, panel=panel)
       fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)
       fig.plot(data=megan_xyz,region=region,projection=proj,style='c0.01i', fill='black',panel=panel)
       fig.plot(data='./for_sia_160725.dat',region=region,projection=proj,style='c0.01i', fill='darkgreen',panel=panel)

       # c       
       panel=[1,0] 
       fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
       gmtsph_grd=pygmt.sph2grd(data='./for_sia_code.sph', spacing=1, region=region)
       fig.grdimage(grid=gmtsph_grd,region=region,projection=proj,frame=frame, panel=panel)
       fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)
       fig.plot(data='./for_sia_160725.dat',region=region,projection=proj,style='c0.01i', fill='darkgreen',panel=panel)

       # d       
       panel=[1,1] 
       fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
       fig.grdimage(grid=propmat_grid_global.to_xarray(),region=region,projection=proj,frame=frame, panel=panel)
       fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)
       fig.plot(data='./for_sia_160725.dat',region=region,projection=proj,style='c0.01i', fill='darkgreen',panel=panel)
       fig.colorbar(region=region,projection=proj,frame=["x+lAmplitude, km"], panel=panel)

       # e       
       panel=[2,0] 
       fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
       fig.grdimage(grid=sh_global_grid.to_xarray(),region=region,projection=proj,frame=frame, panel=panel)
       fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)
       fig.plot(data='./for_sia_160725.dat',region=region,projection=proj,style='c0.01i', fill='darkgreen',panel=panel)

    fig.show()

    return shtools_sh


def lsq_inversion(fname, lmax, norm=1, csphase=1):
    """
    Perform least squares inversion of irregularly sampled function to obtain spherical harmonic coefficients 
    
    Parameters:
    -----------
    file : str
        Filename for the input data containing scattered point data (d, lat, lon).
        
    lmax : int
        Maximum spherical harmonic degree for the expansion.
        
    norm : optional, integer, default = 1
        1 (default) = Geodesy 4-pi normalized harmonics; 2 = Schmidt semi-normalized harmonics; 
        3 = unnormalized harmonics; 4 = orthonormal harmonics.
        
   csphase : optional, integer, default = 1
        1 (default) = do not apply the Condon-Shortley phase factor to the associated 
        Legendre functions; -1 = append the Condon-Shortley phase factor of (-1)^m to the 
        associated Legendre functions.

    Returns:
    --------
    cilm:  float, dimension (2, lmax+1, lmax+1)
        The real spherical harmonic coefficients of the function. 
        The coefficients C0lm and C1lm refer to the cosine (Clm) and sine (Slm) coefficients,
        respectively, with Clm=cilm[0,l,m] and Slm=cilm[1,l,m].
     
    chi2 : float
        The residual sum of squares misfit.    
    """  

    lon, lat, val = fname["lon"], fname["lat"], fname["z"]

    print('performing least squares inversion')
    cilm_b = pysh.SHCoeffs.from_least_squares(val, lat, lon, lmax)

    print('outputting coefficients ready for insertion into propmat')
    new_coeffs = cilm_b.convert(csphase=1, lmax=50, normalization='4pi') * np.sqrt(4*np.pi)
    new_coeffs_arr = pysh.SHCoeffs.to_array(new_coeffs) 
    pysh.shio.shwrite(filename='for_sia_code.sph', coeffs=new_coeffs_arr, lmax=50)
    cmd = " awk -F',' '{print $3, $4}' < for_sia_code.sph > holdt22_lmax40.lm "
    output = sb.check_output(cmd, stderr=sb.STDOUT, shell=True)
    sys.stdout.write('{}'.format(output))

    return cilm_b


def convert_grid_to_sph(fname, lmax, grid_type='DH', sampling=2, k_neighbors=4):
    """
    Convert scattered point data to spherical harmonic representation.

    This function performs inverse distance weighted interpolation on scattered
    data points to create a regular grid, then expands the grid into spherical
    harmonic coefficients.

    Parameters:
    -----------
    fname : str
        Filename for the input data file containing scattered point data
    lmax : int
        Maximum spherical harmonic degree for the expansion
    grid_type : str, optional
        Grid type for pyshtools ('DH' for Driscoll-Healy, 'GLQ' for Gauss-Legendre)
        Default: 'DH'
    sampling : int, optional
        Oversampling factor for the grid (1 or 2)
        Default: 2
    k_neighbors : int, optional
        Number of nearest neighbors to use for interpolation
        Default: 4

    Returns:
    --------
    shtools_sh : pysh.SHCoeffs
        Spherical harmonic coefficients object
    """

    # Load scattered data points: longitude, latitude, and values
    # Data format: space-separated columns (lon, lat, val)
    lon, lat, val = np.loadtxt(
        fname, delimiter=" ", unpack=True, dtype=np.float32)

    # Initialize a pyshtools grid object with specified parameters
    # - lmax: maximum spherical harmonic degree (from parameter)
    # - grid: sampling scheme (configurable)
    # - sampling: The longitudinal sampling for Driscoll and Healy grids. Either 1 for equally sampled grids (nlon = nlat) or 2 for equally spaced grids in degrees
    # - empty=True: create empty grid for later population
    shtools_grid = pysh.SHGrid.from_zeros(
        lmax=lmax, grid=grid_type, sampling=sampling, empty=True)
    shtools_grid.info()

    # Create coordinate meshgrids for the target grid points
    # This generates all longitude-latitude combinations for interpolation
    lons_x, lats_x = np.meshgrid(shtools_grid.lons(), shtools_grid.lats())

    # Convert longitude range from [0, 360] to [-180, 180] for consistency
    lons_x[lons_x > 180] -= 360

    # Build spatial index tree for efficient nearest neighbor search
    # Uses cKDTree for fast spatial queries on scattered data points
    interpolation_tree = cKDTree(np.column_stack((lon, lat)))

    # Query k nearest neighbors for each grid point
    # Returns distances and indices to nearest neighbors
    dists, inds = interpolation_tree.query(
        np.column_stack((lons_x.flatten(), lats_x.flatten())), k=k_neighbors)

    # Create inverse distance weights for interpolation
    # Add small epsilon to prevent division by zero for coincident points
    eps = 1e-10
    weights = 1.0 / (dists + eps)

    # Normalize weights so they sum to 1 for each interpolation point
    # This ensures the interpolation is a proper weighted average
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    # Extract values at the k nearest neighbors for each grid point
    neighbor_values = val[inds]

    # Perform weighted interpolation using optimized einsum operation
    # 'ij,ij->i' notation means: for each point i, sum over neighbors j
    # the element-wise product of weights and values
    interpolated_values = np.einsum('ij,ij->i', weights, neighbor_values)

    # Reshape the flattened interpolated values back to the original grid shape
    shtools_grid.data = interpolated_values.reshape(lons_x.shape)

    # Expand the interpolated grid into spherical harmonic coefficients
    # - normalization='ortho': orthonormalized spherical harmonics
    # - csphase=1: Condon-Shortley phase convention
    shtools_sh = shtools_grid.expand(normalization='4pi', csphase=1)
    
    return shtools_sh


def w_propmat_sph_pyshtools(fname, header, clm):
    """
    Write spherical harmonic coefficients in SPHCO format.

    This function writes spherical harmonic coefficients to a text
    file in the SPHCO format used by geoid processing codes.

    Parameters:
    -----------
    fname : str
        Output filename for the coefficient file
    header : str
        Header string describing the data content
    clm : pyshtools.SHCoeffs
        Spherical harmonic coefficients object with .lmax property and
        .coeffs array of shape (2, lmax+1, lmax+1)

    File Format:
    ------------
    # lmax = <maximum_degree>
    # <header_string>
    #       l   m      Cnm(m>=0)       Snm(m<0)
    SPHCO <degree> <order> <cosine_coeff> <sine_coeff>
    ...

    Notes:
    ------
    - Cosine coefficients (Cnm) are stored in clm.coeffs[0, :, :]
    - Sine coefficients (Snm) are stored in clm.coeffs[1, :, :]
    - Output uses triangular indexing: for each degree, order ranges 0 to degree
    """

    # Extract maximum degree from coefficients object
    lmax = clm.lmax

    # Build all content as a list for efficient string operations
    # This approach minimizes file I/O overhead
    content = []

    # Write file header with metadata
    content.append(f'# lmax = {lmax:2d}\n')
    content.append(f'# {header}\n')
    content.append('#       l   m      Cnm(m>=0)       Snm(m<0)\n')

    # Pre-compute (degree, order) index pairs to eliminate nested loops
    # This replaces the original double loop structure
    degree_order_pairs = [(degree, order)
                          for degree in range(lmax + 1)
                          for order in range(degree + 1)]

    # Extract coefficient arrays once for vectorized access
    # Shape: (lmax+1, lmax+1) for each component
    cosine_coeffs = clm.coeffs[0]  # Cnm coefficients
    sine_coeffs = clm.coeffs[1]    # Snm coefficients

    # Generate all coefficient lines using vectorized formatting
    # Each line follows the SPHCO format with fixed-width fields
    coeff_lines = [f'SPHCO {degree:3d} {order:3d} {cosine_coeffs[degree, order]:14.6e} '
                   f'{sine_coeffs[degree, order]:14.6e}\n'
                   for degree, order in degree_order_pairs]

    # Add all coefficient lines to content
    content.extend(coeff_lines)

    # Write all content to file in a single operation
    # Using context manager ensures proper file closure and error handling
    with open(fname, 'w') as output_file:
        output_file.writelines(content)


def write_fortran_sph_format(fname, header, clm, write_std_dev=True):
    """
    Write spherical harmonic coefficients in Fortran-compatible format.

    This function writes coefficients in the format expected by the Fortran
    code that reads coefficients with the specific order: for m=0 read one
    value, for m>0 read two values (negative and positive m).

    Parameters:
    -----------
    fname : str
        Output filename for the coefficient file
    header : str
        Header string describing the data content
    clm : pyshtools.SHCoeffs
        Spherical harmonic coefficients object
    write_std_dev : bool, optional
        Whether to write standard deviation column (default: True)

    File Format:
    ------------
    # <header_string>
       l    m     C     Standard_Deviation
       0    0   C00    std_dev
       1   -1   C1-1   std_dev
       1    0   C10    std_dev
       1    1   C11    std_dev
       2   -2   C2-2   std_dev
       ...

    Notes:
    ------
    - Coefficients are written in triangular order (l=0 to lmax, m=-l to +l)
    - Only cosine coefficients are written (sine coefficients are ignored)
    - Standard deviations are set to 0.0 if not provided
    """

    # Extract maximum degree and coefficient arrays
    lmax = clm.lmax
    cosine_coeffs = clm.coeffs[0]  # Only use cosine coefficients

    # Build content list for efficient writing
    content = []

    # Write header and column labels
    content.append(f"# {header}\n")
    content.append("   l    m \tC\tStandard_Deviation\n")

    # Generate coefficients in the order expected by Fortran code
    for degree in range(lmax + 1):
        for order in range(-degree, degree + 1):
            # Get coefficient value (use absolute order for array indexing)
            abs_order = abs(order)

            if order >= 0:
                # Positive m: use cosine coefficient directly
                coeff_value = cosine_coeffs[degree, abs_order]
            else:
                # Negative m: use sine coefficient as cosine for negative order
                # This matches the Fortran convention
                coeff_value = clm.coeffs[1, degree, abs_order]

            # Standard deviation (set to 0.0 if not writing std dev)
            std_dev = 0.0 if not write_std_dev else 0.0

            # Format line with proper spacing to match original format
            content.append(
                f"{degree:4d} {order:4d} {coeff_value:12.8f} {std_dev:12.8f}\n")

    # Write all content to file
    with open(fname, 'w') as output_file:
        output_file.writelines(content)


def read_fortran_sph_format(fname):
    """
    Read spherical harmonic coefficients from Fortran-compatible format.

    This function reads coefficients from files written in the format that
    matches the Fortran reading pattern: ordered by degree then by order
    from -l to +l, with only cosine-type coefficients.

    Parameters:
    -----------
    fname : str
        Input filename containing coefficient data

    Returns:
    --------
    clm : pyshtools.SHCoeffs
        Spherical harmonic coefficients object
    header : str
        Header string from file
    lmax : int
        Maximum degree found in file

    File Format Expected:
    ---------------------
    # <header_string>
       l    m     C     Standard_Deviation
       0    0   C00    std_dev
       1   -1   C1-1   std_dev
       1    0   C10    std_dev
       1    1   C11    std_dev
       ...
    """

    # Read all lines from file
    with open(fname, 'r') as input_file:
        lines = input_file.readlines()

    # Parse header and data
    header = ""
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        elif line.startswith('#'):
            header = line[1:].strip()
        elif line.startswith('l ') or line.startswith('   l'):
            continue  # Skip column header
        else:
            data_lines.append(line)

    # Parse coefficient data
    degree_list = []
    order_list = []
    coeff_list = []

    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            degree = int(parts[0])
            order = int(parts[1])
            coeff = float(parts[2])

            degree_list.append(degree)
            order_list.append(order)
            coeff_list.append(coeff)

    # Determine lmax
    lmax = max(degree_list) if degree_list else 0

    # Initialize coefficient arrays
    coeffs = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)

    # Assign coefficients following the Fortran convention
    for degree, order, coeff in zip(degree_list, order_list, coeff_list):
        abs_order = abs(order)

        if order >= 0:
            # Positive m: store as cosine coefficient
            coeffs[0, degree, abs_order] = coeff
        else:
            # Negative m: store as sine coefficient
            coeffs[1, degree, abs_order] = coeff

    # Create pyshtools SHCoeffs object
    clm = pysh.SHCoeffs.from_array(coeffs, normalization='4pi', csphase=1)

    return clm, header, lmax


def read_propmat_sph_pyshtools(fname):
    """
    Read spherical harmonic coefficients from SPHCO format file.

    This function reads spherical harmonic coefficients from a text file
    written in the SPHCO format by w_propmat_sph_pyshtools. It reconstructs
    the pyshtools SHCoeffs object from the file data.

    Parameters:
    -----------
    fname : str
        Input filename containing SPHCO format coefficients

    Returns:
    --------
    clm : pyshtools.SHCoeffs
        Spherical harmonic coefficients object
    header : str
        Header string from the file describing the data content
    lmax : int
        Maximum spherical harmonic degree read from file

    File Format Expected:
    ---------------------
    # lmax = <maximum_degree>
    # <header_string>
    #       l   m      Cnm(m>=0)       Snm(m<0)
    SPHCO <degree> <order> <cosine_coeff> <sine_coeff>
    ...

    Notes:
    ------
    - File must follow the exact SPHCO format specification
    - Coefficients are stored with triangular indexing
    - Missing coefficients are assumed to be zero
    """

    # Read all lines from the file
    with open(fname, 'r') as input_file:
        lines = input_file.readlines()

    # Parse header information
    lmax = None
    header = ""
    coefficient_lines = []

    # Process each line to extract metadata and coefficients
    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Parse lmax from header
        if line.startswith('# lmax ='):
            lmax = int(line.split('=')[1].strip())

        # Parse header description (second comment line)
        elif line.startswith('#') and not line.startswith('# lmax') and not 'Cnm' in line:
            header = line[1:].strip()  # Remove '# ' prefix

        # Collect coefficient data lines
        elif line.startswith('SPHCO'):
            coefficient_lines.append(line)

    # Validate that lmax was found
    if lmax is None:
        raise ValueError(f"Could not find lmax specification in file {fname}")

    # Initialize coefficient arrays
    # Shape: (2, lmax+1, lmax+1) for cosine and sine components
    coeffs = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)

    # Parse coefficient data using vectorized operations where possible
    # Extract degree, order, cosine, and sine values from all lines at once
    degree_list = []
    order_list = []
    cosine_list = []
    sine_list = []

    # Parse all coefficient lines
    for line in coefficient_lines:
        parts = line.split()
        if len(parts) >= 5:  # SPHCO degree order cosine sine
            degree = int(parts[1])
            order = int(parts[2])
            cosine_coeff = float(parts[3])
            sine_coeff = float(parts[4])

            degree_list.append(degree)
            order_list.append(order)
            cosine_list.append(cosine_coeff)
            sine_list.append(sine_coeff)

    # Convert to numpy arrays for vectorized assignment
    degrees = np.array(degree_list)
    orders = np.array(order_list)
    cosines = np.array(cosine_list)
    sines = np.array(sine_list)

    # Vectorized assignment of coefficients
    # coeffs[0] stores cosine coefficients (Cnm)
    # coeffs[1] stores sine coefficients (Snm)
    coeffs[0, degrees, orders] = cosines
    coeffs[1, degrees, orders] = sines

    # Create pyshtools SHCoeffs object
    # Using default normalization and phase conventions
    clm = pysh.SHCoeffs.from_array(coeffs, normalization='4pi', csphase=1)

    return clm, header, lmax


if __name__ == "__main__":
    # Execute main function when script is run directly
    main()
