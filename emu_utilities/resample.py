"""
Code adapted from the ecco_v4_py package at https://ecco-v4-python-tutorial.readthedocs.io/.

Copyright 2018 Ian Fenty

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import division, print_function

import warnings
from collections import OrderedDict

import numpy as np
import pyresample as pr
import xarray as xr
from numpy.typing import NDArray
from xmitgcm.variables import dimensions


def resample_to_latlon(
    orig_lons,
    orig_lats,
    orig_field,
    new_grid_min_lat,
    new_grid_max_lat,
    new_grid_delta_lat,
    new_grid_min_lon,
    new_grid_max_lon,
    new_grid_delta_lon,
    radius_of_influence=120000,
    fill_value=None,
    mapping_method="bin_average",
    neighbors=9,
):
    """Take a field from a source grid and interpolate to a target grid.

    Parameters
    ----------
    orig_lons, orig_lats, orig_field : xarray DataArray or numpy array  :
        the lons, lats, and field from the source grid

        new_grid_min_lat, new_grid_max_lat : float
                latitude limits of new lat-lon grid

    new_grid_delta_lat : float
        latitudinal extent of new lat-lon grid cells in degrees (-90..90)

    new_grid_min_lon, new_grid_max_lon : float
                longitude limits of new lat-lon grid (-180..180)

    new_grid_delta_lon : float
         longitudinal extent of new lat-lon grid cells in degrees

    radius_of_influence : float, optional.  Default 120000 m
        the radius of the circle within which data from the
        original field (orig_field) is used when mapping to the new grid

    fill_value : float, optional. Default None
                value to use in the new lat-lon grid if there are no valid values
                from the source grid

        mapping_method : string, optional. Default 'bin_average'
        denote the type of interpolation method to use.
        options include
            'nearest_neighbor' - Take the nearest value from the source grid
                                                     to the target grid
            'bin_average'      - Use the average value from the source grid
                                                                 to the target grid

    neighbors : int, optional. Default 9
        from pyresample ("neighbours" parameter, note English alternative spelling)
        The maximum number of neigbors on the original field (orig_field)
        to use when mapping the original field to the new grid.
        If bin-averaging, pyresample will only include up to 'neighbors'
        number of closest points. Setting this number higher increases memory
        usage. see pyresample for me information

    RETURNS:
    new_grid_lon_centers, new_grid_lat_centers : ndarrays
        2D arrays with the lon and lat values of the new grid cell centers

    new_grid_lon_edges, new_grid_lat_edges: ndarrays
        2D arrays with the lon and lat values of the new grid cell edges

    data_latlon_projection:
        the source field interpolated to the new grid

    """
    if type(orig_lats) is xr.DataArray:
        orig_lons_1d = orig_lons.values.ravel()
        orig_lats_1d = orig_lats.values.ravel()

    elif type(orig_lats) is np.ndarray:
        orig_lats_1d = orig_lats.ravel()
        orig_lons_1d = orig_lons.ravel()
    else:
        raise TypeError(
            "orig_lons and orig_lats variable either a DataArray or numpy.ndarray. \n"
            "Found type(orig_lons) = %s and type(orig_lats) = %s" % (type(orig_lons), type(orig_lats))
        )

    if type(orig_field) is xr.DataArray:
        orig_field = orig_field.values
    elif type(orig_field) is not np.ndarray and type(orig_field) is not np.ma.MaskedArray:
        raise TypeError(
            "orig_field must be a type of DataArray, ndarray, or MaskedArray. \n"
            "Found type(orig_field) = %s" % type(orig_field)
        )
    ## Modifications to allow time and depth dimensions (DS, 2023-04-20)
    # Collapse any non-horizontal dimensions into a single, final dimension:

    # Get shape of orig_lats, then difference with orig_field
    n_horiz_dims = len(orig_lats.shape)
    n_total_dims = len(orig_field.shape)
    n_extra_dims = n_total_dims - n_horiz_dims
    horiz_dim_shape = orig_lats.shape  # e.g. [13,90,90]
    if (n_extra_dims > 0) & (np.prod(orig_field.shape) > np.prod(horiz_dim_shape)):
        # If there are extra dimensions (and they are meaningful/have len > 1)...

        # Check if extra dimensions are at beginning or end of orig_field...
        if orig_field.shape[0] != orig_lats.shape[0]:
            # ... if at the beginning, collapse and move to end
            extra_dims_at_beginning = True
            extra_dim_shape = orig_field.shape[:n_extra_dims]  # e.g. [312,50]
            new_shape = np.hstack(
                [np.prod(extra_dim_shape), np.prod(horiz_dim_shape)]
            )  # e.g. from [312,50,13,90,90] to [15600,105300]
            orig_field = orig_field.reshape(new_shape).transpose(1, 0)  # e.g. from [15600,105300] to [105300,15600]
        else:
            # ... if at the end, just collapse
            extra_dims_at_beginning = False
            extra_dim_shape = orig_field.shape[n_horiz_dims:]  # e.g. [50,312]
            new_shape = np.hstack(
                [np.prod(horiz_dim_shape), np.prod(extra_dim_shape)]
            )  # e.g. from [13,90,90,50,312] to [105300,15600]
            orig_field = orig_field.reshape(new_shape)
    ##

    # prepare for the nearest neighbor mapping

    # first define the lat lon points of the original data
    orig_grid = pr.geometry.SwathDefinition(lons=orig_lons_1d, lats=orig_lats_1d)

    # the latitudes to which we will we interpolate
    num_lats = int((new_grid_max_lat - new_grid_min_lat) / new_grid_delta_lat + 1)
    num_lons = int((new_grid_max_lon - new_grid_min_lon) / new_grid_delta_lon + 1)

    if (num_lats > 0) and (num_lons > 0):
        # linspace is preferred when using floats!

        new_grid_lat_edges_1D = np.linspace(new_grid_min_lat, new_grid_max_lat, num=int(num_lats))

        new_grid_lon_edges_1D = np.linspace(new_grid_min_lon, new_grid_max_lon, num=int(num_lons))

        new_grid_lat_centers_1D = (new_grid_lat_edges_1D[0:-1] + new_grid_lat_edges_1D[1:]) / 2
        new_grid_lon_centers_1D = (new_grid_lon_edges_1D[0:-1] + new_grid_lon_edges_1D[1:]) / 2

        new_grid_lon_edges, new_grid_lat_edges = np.meshgrid(new_grid_lon_edges_1D, new_grid_lat_edges_1D)

        new_grid_lon_centers, new_grid_lat_centers = np.meshgrid(new_grid_lon_centers_1D, new_grid_lat_centers_1D)

        # print(np.min(new_grid_lon_centers), np.max(new_grid_lon_centers))
        # print(np.min(new_grid_lon_edges), np.max(new_grid_lon_edges))

        # print(np.min(new_grid_lat_centers), np.max(new_grid_lat_centers))
        # print(np.min(new_grid_lat_edges), np.max(new_grid_lat_edges))

        # define the lat lon points of the two parts.
        new_grid = pr.geometry.GridDefinition(lons=new_grid_lon_centers, lats=new_grid_lat_centers)

        if mapping_method == "nearest_neighbor":
            data_latlon_projection = pr.kd_tree.resample_nearest(
                orig_grid, orig_field, new_grid, radius_of_influence=radius_of_influence, fill_value=fill_value
            )
        elif mapping_method == "bin_average":
            data_latlon_projection = pr.kd_tree.resample_custom(
                orig_grid,
                orig_field,
                new_grid,
                radius_of_influence=radius_of_influence,
                weight_funcs=lambda r: 1,
                fill_value=fill_value,
                neighbours=neighbors,
            )
        else:
            raise ValueError(
                "mapping_method must be nearest_neighbor or bin_average. \nFound mapping_method = %s " % mapping_method
            )

        ## Modifications to allow time and depth dimensions (DS, 2023-04-20)
        if (n_extra_dims > 0) & (np.prod(orig_field.shape) > np.prod(horiz_dim_shape)):
            # If there are extra dimensions (and they are meaningful/have len > 1)
            new_horiz_shape = data_latlon_projection.shape[:2]
            if extra_dims_at_beginning:
                # If the extra dimensions were originally at the beginning, move back...
                data_latlon_projection = data_latlon_projection.transpose(2, 0, 1)
                # ... and unstack the additional dimensions
                final_shape = np.hstack([extra_dim_shape, new_horiz_shape])
                data_latlon_projection = data_latlon_projection.reshape(final_shape)
            else:
                # If the extra dimensions were originally at the end, just unstack
                final_shape = np.hstack([extra_dim_shape, new_horiz_shape])
                data_latlon_projection = data_latlon_projection.reshape(final_shape)
        ##

    else:
        raise ValueError(
            "Number of lat and lon points to interpolate to must be > 0. \n"
            "Found num_lats = %d, num lons = %d" % (num_lats, num_lons)
        )

    return new_grid_lon_centers, new_grid_lat_centers, new_grid_lon_edges, new_grid_lat_edges, data_latlon_projection


def resample_ds(
    ds,
    new_grid_min_lat,
    new_grid_max_lat,
    new_grid_delta_lat,
    new_grid_min_lon,
    new_grid_max_lon,
    new_grid_delta_lon,
    radius_of_influence=120000,
    fill_value=None,
    mapping_method="bin_average",
    neighbors=9,
):
    """Take a field from a source grid and interpolate to a target grid.

    Parameters
    ----------
    orig_lons, orig_lats, orig_field : xarray DataArray or numpy array  :
        the lons, lats, and field from the source grid

        new_grid_min_lat, new_grid_max_lat : float
                latitude limits of new lat-lon grid

    new_grid_delta_lat : float
        latitudinal extent of new lat-lon grid cells in degrees (-90..90)

    new_grid_min_lon, new_grid_max_lon : float
                longitude limits of new lat-lon grid (-180..180)

    new_grid_delta_lon : float
         longitudinal extent of new lat-lon grid cells in degrees

    radius_of_influence : float, optional.  Default 120000 m
        the radius of the circle within which data from the
        original field (orig_field) is used when mapping to the new grid

    fill_value : float, optional. Default None
                value to use in the new lat-lon grid if there are no valid values
                from the source grid

        mapping_method : string, optional. Default 'bin_average'
        denote the type of interpolation method to use.
        options include
            'nearest_neighbor' - Take the nearest value from the source grid
                                                     to the target grid
            'bin_average'      - Use the average value from the source grid
                                                                 to the target grid

    neighbors : int, optional. Default 9
        from pyresample ("neighbours" parameter, note English alternative spelling)
        The maximum number of neigbors on the original field (orig_field)
        to use when mapping the original field to the new grid.
        If bin-averaging, pyresample will only include up to 'neighbors'
        number of closest points. Setting this number higher increases memory
        usage. see pyresample for me information

    RETURNS:
    new_grid_lon_centers, new_grid_lat_centers : ndarrays
        2D arrays with the lon and lat values of the new grid cell centers

    new_grid_lon_edges, new_grid_lat_edges: ndarrays
        2D arrays with the lon and lat values of the new grid cell edges

    data_latlon_projection:
        the source field interpolated to the new grid

    """
    data_vars = ds.keys()
    ds_out = ds.copy(deep=True)
    for data_var in data_vars:
        if (
            "tile" not in ds[data_var].dims
            or ("j" not in ds[data_var].dims and "j_g" not in ds[data_var].dims)
            or ("i" not in ds[data_var].dims and "i_g" not in ds[data_var].dims)
        ):
            continue  # skip variables that don't have the right dimensions
            warnings.warn(f"Skipping variable {data_var} as it does not have the required dimensions.")
        if "i_g" in ds[data_var].dims:
            orig_lons = ds[data_var].xg.values
        else:
            orig_lons = ds[data_var].xc.values
        if "j_g" in ds[data_var].dims:
            orig_lats = ds[data_var].yg.values
        else:
            orig_lats = ds[data_var].yc.values
        orig_field = ds[data_var].values

        orig_lats_1d = orig_lats.ravel()
        orig_lons_1d = orig_lons.ravel()
        ## Modifications to allow time and depth dimensions (DS, 2023-04-20)
        # Collapse any non-horizontal dimensions into a single, final dimension:

        # Get shape of orig_lats, then difference with orig_field
        n_horiz_dims = len(orig_lats.shape)
        n_total_dims = len(orig_field.shape)
        n_extra_dims = n_total_dims - n_horiz_dims
        horiz_dim_shape = orig_lats.shape  # e.g. [13,90,90]
        if (n_extra_dims > 0) & (np.prod(orig_field.shape) > np.prod(horiz_dim_shape)):
            # If there are extra dimensions (and they are meaningful/have len > 1)...

            # Check if extra dimensions are at beginning or end of orig_field...
            if orig_field.shape[0] != orig_lats.shape[0]:
                # ... if at the beginning, collapse and move to end
                extra_dims_at_beginning = True
                extra_dim_shape = orig_field.shape[:n_extra_dims]  # e.g. [312,50]
                new_shape = np.hstack(
                    [np.prod(extra_dim_shape), np.prod(horiz_dim_shape)]
                )  # e.g. from [312,50,13,90,90] to [15600,105300]
                orig_field = orig_field.reshape(new_shape).transpose(1, 0)  # e.g. from [15600,105300] to [105300,15600]
            else:
                # ... if at the end, just collapse
                extra_dims_at_beginning = False
                extra_dim_shape = orig_field.shape[n_horiz_dims:]  # e.g. [50,312]
                new_shape = np.hstack(
                    [np.prod(horiz_dim_shape), np.prod(extra_dim_shape)]
                )  # e.g. from [13,90,90,50,312] to [105300,15600]
                orig_field = orig_field.reshape(new_shape)
        ##

        # prepare for the nearest neighbor mapping

        # first define the lat lon points of the original data
        orig_grid = pr.geometry.SwathDefinition(lons=orig_lons_1d, lats=orig_lats_1d)

        # the latitudes to which we will we interpolate
        num_lats = int((new_grid_max_lat - new_grid_min_lat) / new_grid_delta_lat + 1)
        num_lons = int((new_grid_max_lon - new_grid_min_lon) / new_grid_delta_lon + 1)

        if (num_lats > 0) and (num_lons > 0):
            # linspace is preferred when using floats!

            new_grid_lat_edges_1D = np.linspace(new_grid_min_lat, new_grid_max_lat, num=int(num_lats))

            new_grid_lon_edges_1D = np.linspace(new_grid_min_lon, new_grid_max_lon, num=int(num_lons))

            new_grid_lat_centers_1D = (new_grid_lat_edges_1D[0:-1] + new_grid_lat_edges_1D[1:]) / 2
            new_grid_lon_centers_1D = (new_grid_lon_edges_1D[0:-1] + new_grid_lon_edges_1D[1:]) / 2

            new_grid_lon_edges, new_grid_lat_edges = np.meshgrid(new_grid_lon_edges_1D, new_grid_lat_edges_1D)

            new_grid_lon_centers, new_grid_lat_centers = np.meshgrid(new_grid_lon_centers_1D, new_grid_lat_centers_1D)

            # print(np.min(new_grid_lon_centers), np.max(new_grid_lon_centers))
            # print(np.min(new_grid_lon_edges), np.max(new_grid_lon_edges))

            # print(np.min(new_grid_lat_centers), np.max(new_grid_lat_centers))
            # print(np.min(new_grid_lat_edges), np.max(new_grid_lat_edges))

            # define the lat lon points of the two parts.
            new_grid = pr.geometry.GridDefinition(lons=new_grid_lon_centers, lats=new_grid_lat_centers)

            if mapping_method == "nearest_neighbor":
                data_latlon_projection = pr.kd_tree.resample_nearest(
                    orig_grid, orig_field, new_grid, radius_of_influence=radius_of_influence, fill_value=fill_value
                )
            elif mapping_method == "bin_average":
                data_latlon_projection = pr.kd_tree.resample_custom(
                    orig_grid,
                    orig_field,
                    new_grid,
                    radius_of_influence=radius_of_influence,
                    weight_funcs=lambda r: 1,
                    fill_value=fill_value,
                    neighbours=neighbors,
                )
            else:
                raise ValueError(
                    "mapping_method must be nearest_neighbor or bin_average. \nFound mapping_method = %s "
                    % mapping_method
                )

            ## Modifications to allow time and depth dimensions (DS, 2023-04-20)
            if (n_extra_dims > 0) & (np.prod(orig_field.shape) > np.prod(horiz_dim_shape)):
                # If there are extra dimensions (and they are meaningful/have len > 1)
                new_horiz_shape = data_latlon_projection.shape[:2]
                if extra_dims_at_beginning:
                    # If the extra dimensions were originally at the beginning, move back...
                    data_latlon_projection = data_latlon_projection.transpose(2, 0, 1)
                    # ... and unstack the additional dimensions
                    final_shape = np.hstack([extra_dim_shape, new_horiz_shape])
                    data_latlon_projection = data_latlon_projection.reshape(final_shape)
                else:
                    # If the extra dimensions were originally at the end, just unstack
                    final_shape = np.hstack([extra_dim_shape, new_horiz_shape])
                    data_latlon_projection = data_latlon_projection.reshape(final_shape)
            ##

        else:
            raise ValueError(
                "Number of lat and lon points to interpolate to must be > 0. \n"
                "Found num_lats = %d, num lons = %d" % (num_lats, num_lons)
            )

        coords = {c: ds[data_var][c].values for c in ds[data_var].dims if c not in ["tile", "j", "i", "xc", "yc"]}
        coords["lat"] = new_grid_lat_centers_1D
        coords["lon"] = new_grid_lon_centers_1D

        ds_out[data_var] = xr.DataArray(
            data_latlon_projection,
            coords=coords,
        )

    ds_out = ds_out.drop_vars(["xc", "yc"])
    ds_out = ds_out.drop_dims(["tile", "j", "i"])

    return ds_out


def llc_compact_to_tiles(data_compact, less_output=True):
    """

    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.

    Parameters
    ----------
    data_compact : ndarray
        a numpy array of dimension nl x nk x 13*llc x llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output


    Returns
    -------
    data_tiles : ndarray
        a numpy array organized by, at most,
        13 tiles x nl x nk x llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions in data_tiles

    """

    data_tiles = llc_faces_to_tiles(
        llc_compact_to_faces(data_compact, less_output=less_output), less_output=less_output
    )

    return data_tiles


def llc_tiles_to_compact(data_tiles, less_output=False):
    """

    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.

    Parameters
    ----------
    data_tiles : ndarray
        a numpy array organized by, at most,
        13 tiles x nl x nk x llc x llc

        where dimensions 'nl' and 'nk' are optional.

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_compact : ndarray
        a numpy array of dimension nl x nk x 13*llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions in data_compact

    """

    data_faces = llc_tiles_to_faces(data_tiles, less_output=less_output)
    data_compact = llc_faces_to_compact(data_faces, less_output=less_output)

    return data_compact


def llc_compact_to_faces(data_compact, less_output=False):
    """
    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it into the 5 'faces'
    of the llc grid.

    The five faces are 4 approximately lat-lon oriented and one Arctic 'cap'

    Parameters
    ----------
    data_compact : ndarray
        An 2D array of dimension  nl x nk x 13*llc x llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output


    Returns
    -------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_compact

    """

    dims = data_compact.shape
    num_dims = len(dims)

    # final dimension is always of length llc
    llc = dims[-1]

    # dtype of compact array
    arr_dtype = data_compact.dtype

    if not less_output:
        print("llc_compact_to_faces: dims, llc ", dims, llc)
        print("llc_compact_to_faces: data_compact array type ", data_compact.dtype)

    if num_dims == 2:  # we have a single 2D slices (y, x)
        f1 = np.zeros((3 * llc, llc), dtype=arr_dtype)
        f2 = np.zeros((3 * llc, llc), dtype=arr_dtype)
        f3 = np.zeros((llc, llc), dtype=arr_dtype)
        f4 = np.zeros((llc, 3 * llc), dtype=arr_dtype)
        f5 = np.zeros((llc, 3 * llc), dtype=arr_dtype)

    elif num_dims == 3:  # we have 3D slices (time or depth, y, x)
        nk = dims[0]

        f1 = np.zeros((nk, 3 * llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nk, 3 * llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nk, llc, 3 * llc), dtype=arr_dtype)
        f5 = np.zeros((nk, llc, 3 * llc), dtype=arr_dtype)

    elif num_dims == 4:  # we have a 4D slice (time or depth, time or depth, y, x)
        nl = dims[0]
        nk = dims[1]

        f1 = np.zeros((nl, nk, 3 * llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nl, nk, 3 * llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nl, nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nl, nk, llc, 3 * llc), dtype=arr_dtype)
        f5 = np.zeros((nl, nk, llc, 3 * llc), dtype=arr_dtype)

    else:
        print("llc_compact_to_faces: can only handle compact arrays of 2, 3, or 4 dimensions!")
        return []

    # map the data from the compact format to the five face arrays

    # -- 2D case
    if num_dims == 2:
        f1 = data_compact[: 3 * llc, :]
        f2 = data_compact[3 * llc : 6 * llc, :]
        f3 = data_compact[6 * llc : 7 * llc, :]

        # f4 = np.zeros((llc, 3*llc))

        for f in range(8, 11):
            i1 = np.arange(0, llc) + (f - 8) * llc
            i2 = np.arange(0, 3 * llc, 3) + 7 * llc + f - 8
            f4[:, i1] = data_compact[i2, :]

        # f5 = np.zeros((llc, 3*llc))

        for f in range(11, 14):
            i1 = np.arange(0, llc) + (f - 11) * llc
            i2 = np.arange(0, 3 * llc, 3) + 10 * llc + f - 11
            # print ('f, i1, i2 ', f, i1[0], i2[0])

            f5[:, i1] = data_compact[i2, :]

    # -- 3D case
    elif num_dims == 3:
        # loop over k

        for k in range(nk):
            f1[k, :] = data_compact[k, : 3 * llc, :]
            f2[k, :] = data_compact[k, 3 * llc : 6 * llc, :]
            f3[k, :] = data_compact[k, 6 * llc : 7 * llc, :]

            # if someone could explain why I have to make
            # dummy arrays of f4_tmp and f5_tmp instead of just using
            # f5 directly I would be so grateful!
            f4_tmp = np.zeros((llc, 3 * llc))
            f5_tmp = np.zeros((llc, 3 * llc))

            for f in range(8, 11):
                i1 = np.arange(0, llc) + (f - 8) * llc
                i2 = np.arange(0, 3 * llc, 3) + 7 * llc + f - 8
                f4_tmp[:, i1] = data_compact[k, i2, :]

            for f in range(11, 14):
                i1 = np.arange(0, llc) + (f - 11) * llc
                i2 = np.arange(0, 3 * llc, 3) + 10 * llc + f - 11
                f5_tmp[:, i1] = data_compact[k, i2, :]

            f4[k, :] = f4_tmp
            f5[k, :] = f5_tmp

    # -- 4D case
    elif num_dims == 4:
        # loop over l and k
        for l in range(nl):
            for k in range(nk):
                f1[l, k, :] = data_compact[l, k, : 3 * llc, :]
                f2[l, k, :] = data_compact[l, k, 3 * llc : 6 * llc, :]
                f3[l, k, :] = data_compact[l, k, 6 * llc : 7 * llc, :]

                # if someone could explain why I have to make
                # dummy arrays of f4_tmp and f5_tmp instead of just using
                # f5 directly I would be so grateful!
                f4_tmp = np.zeros((llc, 3 * llc))
                f5_tmp = np.zeros((llc, 3 * llc))

                for f in range(8, 11):
                    i1 = np.arange(0, llc) + (f - 8) * llc
                    i2 = np.arange(0, 3 * llc, 3) + 7 * llc + f - 8
                    f4_tmp[:, i1] = data_compact[l, k, i2, :]

                for f in range(11, 14):
                    i1 = np.arange(0, llc) + (f - 11) * llc
                    i2 = np.arange(0, 3 * llc, 3) + 10 * llc + f - 11
                    f5_tmp[:, i1] = data_compact[l, k, i2, :]

                f4[l, k, :, :] = f4_tmp
                f5[l, k, :, :] = f5_tmp

    # put the 5 faces in the dictionary.
    F = {}
    F[1] = f1
    F[2] = f2
    F[3] = f3
    F[4] = f4
    F[5] = f5

    return F


def llc_faces_to_tiles(F, less_output=False) -> NDArray:
    """

    Converts a dictionary, F, containing 5 lat-lon-cap faces into 13 tiles
    of dimension nl x nk x llc x llc x nk.

    Tiles 1-6 and 8-13 are oriented approximately lat-lon
    while tile 7 is the Arctic 'cap'

    Parameters
    ----------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_tiles : ndarray
        an array of dimension 13 x nl x nk x llc x llc,

        Each 2D slice is dimension 13 x llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_tiles


    """

    # pull out the five face arrays
    f1 = F[1]
    f2 = F[2]
    f3 = F[3]
    f4 = F[4]
    f5 = F[5]

    dims = f3.shape
    num_dims = len(dims)

    # dtype of compact array
    arr_dtype = f1.dtype

    # final dimension of face 1 is always of length llc
    ni_3 = f3.shape[-1]

    llc = ni_3  # default
    #

    if num_dims == 2:  # we have a single 2D slices (y, x)
        data_tiles = np.zeros((13, llc, llc), dtype=arr_dtype)

    elif num_dims == 3:  # we have 3D slices (time or depth, y, x)
        nk = dims[0]
        data_tiles = np.zeros((nk, 13, llc, llc), dtype=arr_dtype)

    elif num_dims == 4:  # we have a 4D slice (time or depth, time or depth, y, x)
        nl = dims[0]
        nk = dims[1]

        data_tiles = np.zeros((nl, nk, 13, llc, llc), dtype=arr_dtype)

    else:
        print("llc_faces_to_tiles: can only handle face arrays that have 2, 3, or 4 dimensions!")
        return []

    # llc is the length of the second dimension
    if not less_output:
        print("llc_faces_to_tiles: data_tiles shape ", data_tiles.shape)
        print("llc_faces_to_tiles: data_tiles dtype ", data_tiles.dtype)

    # map the data from the faces format to the 13 tile arrays

    # -- 2D case
    if num_dims == 2:
        data_tiles[0, :] = f1[llc * 0 : llc * 1, :]
        data_tiles[1, :] = f1[llc * 1 : llc * 2, :]
        data_tiles[2, :] = f1[llc * 2 :, :]
        data_tiles[3, :] = f2[llc * 0 : llc * 1, :]
        data_tiles[4, :] = f2[llc * 1 : llc * 2, :]
        data_tiles[5, :] = f2[llc * 2 :, :]
        data_tiles[6, :] = f3
        data_tiles[7, :] = f4[:, llc * 0 : llc * 1]
        data_tiles[8, :] = f4[:, llc * 1 : llc * 2]
        data_tiles[9, :] = f4[:, llc * 2 :]
        data_tiles[10, :] = f5[:, llc * 0 : llc * 1]
        data_tiles[11, :] = f5[:, llc * 1 : llc * 2]
        data_tiles[12, :] = f5[:, llc * 2 :]

    # -- 3D case
    if num_dims == 3:
        # loop over k
        for k in range(nk):
            data_tiles[k, 0, :] = f1[k, llc * 0 : llc * 1, :]
            data_tiles[k, 1, :] = f1[k, llc * 1 : llc * 2, :]
            data_tiles[k, 2, :] = f1[k, llc * 2 :, :]
            data_tiles[k, 3, :] = f2[k, llc * 0 : llc * 1, :]
            data_tiles[k, 4, :] = f2[k, llc * 1 : llc * 2, :]
            data_tiles[k, 5, :] = f2[k, llc * 2 :, :]
            data_tiles[k, 6, :] = f3[k, :]
            data_tiles[k, 7, :] = f4[k, :, llc * 0 : llc * 1]
            data_tiles[k, 8, :] = f4[k, :, llc * 1 : llc * 2]
            data_tiles[k, 9, :] = f4[k, :, llc * 2 :]
            data_tiles[k, 10, :] = f5[k, :, llc * 0 : llc * 1]
            data_tiles[k, 11, :] = f5[k, :, llc * 1 : llc * 2]
            data_tiles[k, 12, :] = f5[k, :, llc * 2 :]

    # -- 4D case
    if num_dims == 4:
        # loop over l and k
        for l in range(nl):
            for k in range(nk):
                data_tiles[l, k, 0, :] = f1[l, k, llc * 0 : llc * 1, :]
                data_tiles[l, k, 1, :] = f1[l, k, llc * 1 : llc * 2, :]
                data_tiles[l, k, 2, :] = f1[l, k, llc * 2 :, :]
                data_tiles[l, k, 3, :] = f2[l, k, llc * 0 : llc * 1, :]
                data_tiles[l, k, 4, :] = f2[l, k, llc * 1 : llc * 2, :]
                data_tiles[l, k, 5, :] = f2[l, k, llc * 2 :, :]
                data_tiles[l, k, 6, :] = f3[l, k, :]
                data_tiles[l, k, 7, :] = f4[l, k, :, llc * 0 : llc * 1]
                data_tiles[l, k, 8, :] = f4[l, k, :, llc * 1 : llc * 2]
                data_tiles[l, k, 9, :] = f4[l, k, :, llc * 2 :]
                data_tiles[l, k, 10, :] = f5[l, k, :, llc * 0 : llc * 1]
                data_tiles[l, k, 11, :] = f5[l, k, :, llc * 1 : llc * 2]
                data_tiles[l, k, 12, :] = f5[l, k, :, llc * 2 :]

    return data_tiles


def llc_ig_jg_faces_to_tiles(F, less_output=False):
    """

    Converts a dictionary, F, containing 5 lat-lon-cap faces into 13 tiles
    of dimension nl x nk x llc+1 x llc+1 x nk.

    ig_jg_faces arrays include one extra 'row' and 'column'
    for the 'north' and 'east' points of the array.

    Tiles 1-6 and 8-13 are oriented approximately lat-lon
    while tile 7 is the Arctic 'cap'

    Parameters
    ----------
    F : dict
        a dictionary containing the five lat-lon-cap face arrays that include
        one extra 'row' and 'column' for the 'north' and 'east' points

        F[n] is a numpy array of face n, n in [1..5]

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_tiles : ndarray
        an array of dimension 13 x nl x nk x llc+1 x llc+1,

        Each 2D slice is dimension 13 x llc+1 x llc+1

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_tiles


    """

    # pull out the five face arrays
    f1 = F[1]
    f2 = F[2]
    f3 = F[3]
    f4 = F[4]
    f5 = F[5]

    dims = f3.shape
    num_dims = len(dims)

    # dtype of compact array
    arr_dtype = f1.dtype

    # final dimension of face 1 length llc +1
    ni_3 = f3.shape[-1]

    llc = ni_3 - 1  # 1 is subtracted because array has extra north and east points
    #

    if num_dims == 2:  # we have a single 2D slices (y, x)
        data_tiles = np.zeros((13, llc + 1, llc + 1), dtype=arr_dtype)

    elif num_dims == 3:  # we have 3D slices (time or depth, y, x)
        nk = dims[0]
        data_tiles = np.zeros((nk, 13, llc + 1, llc + 1), dtype=arr_dtype)

    elif num_dims == 4:  # we have a 4D slice (time or depth, time or depth, y, x)
        nl = dims[0]
        nk = dims[1]

        data_tiles = np.zeros((nl, nk, 13, llc + 1, llc + 1), dtype=arr_dtype)

    else:
        print("llc_faces_to_tiles: can only handle face arrays that have 2, 3, or 4 dimensions!")
        return []

    # llc is the length of the second dimension
    if not less_output:
        print("llc_faces_to_tiles: data_tiles shape ", data_tiles.shape)
        print("llc_faces_to_tiles: data_tiles dtype ", data_tiles.dtype)

    # map the data from the faces format to the 13 tile arrays

    # -- 2D case
    if num_dims == 2:
        data_tiles[0, :] = f1[llc * 0 : llc * 1 + 1, :]
        data_tiles[1, :] = f1[llc * 1 : llc * 2 + 1, :]
        data_tiles[2, :] = f1[llc * 2 :, :]
        data_tiles[3, :] = f2[llc * 0 : llc * 1 + 1, :]
        data_tiles[4, :] = f2[llc * 1 : llc * 2 + 1, :]
        data_tiles[5, :] = f2[llc * 2 :, :]
        data_tiles[6, :] = f3
        data_tiles[7, :] = f4[:, llc * 0 : llc * 1 + 1]
        data_tiles[8, :] = f4[:, llc * 1 : llc * 2 + 1]
        data_tiles[9, :] = f4[:, llc * 2 :]
        data_tiles[10, :] = f5[:, llc * 0 : llc * 1 + 1]
        data_tiles[11, :] = f5[:, llc * 1 : llc * 2 + 1]
        data_tiles[12, :] = f5[:, llc * 2 :]

    # -- 3D case
    if num_dims == 3:
        # loop over k
        for k in range(nk):
            data_tiles[k, 0, :] = f1[k, llc * 0 : llc * 1 + 1, :]
            data_tiles[k, 1, :] = f1[k, llc * 1 : llc * 2 + 1, :]
            data_tiles[k, 2, :] = f1[k, llc * 2 :, :]
            data_tiles[k, 3, :] = f2[k, llc * 0 : llc * 1 + 1, :]
            data_tiles[k, 4, :] = f2[k, llc * 1 : llc * 2 + 1, :]
            data_tiles[k, 5, :] = f2[k, llc * 2 :, :]
            data_tiles[k, 6, :] = f3[k, :]
            data_tiles[k, 7, :] = f4[k, :, llc * 0 : llc * 1 + 1]
            data_tiles[k, 8, :] = f4[k, :, llc * 1 : llc * 2 + 1]
            data_tiles[k, 9, :] = f4[k, :, llc * 2 :]
            data_tiles[k, 10, :] = f5[k, :, llc * 0 : llc * 1 + 1]
            data_tiles[k, 11, :] = f5[k, :, llc * 1 : llc * 2 + 1]
            data_tiles[k, 12, :] = f5[k, :, llc * 2 :]

    # -- 4D case
    if num_dims == 4:
        # loop over l and k
        for l in range(nl):
            for k in range(nk):
                data_tiles[l, k, 0, :] = f1[l, k, llc * 0 : llc * 1 + 1, :]
                data_tiles[l, k, 1, :] = f1[l, k, llc * 1 : llc * 2 + 1, :]
                data_tiles[l, k, 2, :] = f1[l, k, llc * 2 :, :]
                data_tiles[l, k, 3, :] = f2[l, k, llc * 0 : llc * 1 + 1, :]
                data_tiles[l, k, 4, :] = f2[l, k, llc * 1 : llc * 2 + 1, :]
                data_tiles[l, k, 5, :] = f2[l, k, llc * 2 :, :]
                data_tiles[l, k, 6, :] = f3[l, k, :]
                data_tiles[l, k, 7, :] = f4[l, k, :, llc * 0 : llc * 1 + 1]
                data_tiles[l, k, 8, :] = f4[l, k, :, llc * 1 : llc * 2 + 1]
                data_tiles[l, k, 9, :] = f4[l, k, :, llc * 2 :]
                data_tiles[l, k, 10, :] = f5[l, k, :, llc * 0 : llc * 1 + 1]
                data_tiles[l, k, 11, :] = f5[l, k, :, llc * 1 : llc * 2 + 1]
                data_tiles[l, k, 12, :] = f5[l, k, :, llc * 2 :]

    return data_tiles


def llc_tiles_to_faces(data_tiles, less_output=False):
    """

    Converts an array of 13 'tiles' from the lat-lon-cap grid
    and rearranges them to 5 faces.  Faces 1,2,4, and 5 are approximately
    lat-lon while face 3 is the Arctic 'cap'

    Parameters
    ----------
    data_tiles :
        An array of dimension 13 x nl x nk x llc x llc

    If dimensions nl or nk are singular, they are not included
        as dimensions of data_tiles

    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False

    Returns
    -------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc

    """

    # ascertain how many dimensions are in the faces (minimum 3, maximum 5)
    dims = data_tiles.shape
    num_dims = len(dims)

    # the final dimension is always length llc
    llc = dims[-1]

    # tiles is always just before (y,x) dims
    num_tiles = dims[-3]

    # data type of original data_tiles
    arr_dtype = data_tiles.dtype

    if not less_output:
        print("llc_tiles_to_faces: num_tiles, ", num_tiles)

    if num_dims == 3:  # we have a 13 2D slices (tile, y, x)
        f1 = np.zeros((3 * llc, llc), dtype=arr_dtype)
        f2 = np.zeros((3 * llc, llc), dtype=arr_dtype)
        f3 = np.zeros((llc, llc), dtype=arr_dtype)
        f4 = np.zeros((llc, 3 * llc), dtype=arr_dtype)
        f5 = np.zeros((llc, 3 * llc), dtype=arr_dtype)

    elif num_dims == 4:  # 13 3D slices (time or depth, tile, y, x)
        nk = dims[0]

        f1 = np.zeros((nk, 3 * llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nk, 3 * llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nk, llc, 3 * llc), dtype=arr_dtype)
        f5 = np.zeros((nk, llc, 3 * llc), dtype=arr_dtype)

    elif num_dims == 5:  # 4D slice (time or depth, time or depth, tile, y, x)
        nl = dims[0]
        nk = dims[1]

        f1 = np.zeros((nl, nk, 3 * llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nl, nk, 3 * llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nl, nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nl, nk, llc, 3 * llc), dtype=arr_dtype)
        f5 = np.zeros((nl, nk, llc, 3 * llc), dtype=arr_dtype)

    else:
        print("llc_tiles_to_faces: can only handle tiles that have 2, 3, or 4 dimensions!")
        return []

    # Map data on the tiles to the faces

    # 2D slices on 13 tiles
    if num_dims == 3:
        f1[llc * 0 : llc * 1, :] = data_tiles[0, :]

        f1[llc * 1 : llc * 2, :] = data_tiles[1, :]
        f1[llc * 2 :, :] = data_tiles[2, :]

        f2[llc * 0 : llc * 1, :] = data_tiles[3, :]
        f2[llc * 1 : llc * 2, :] = data_tiles[4, :]
        f2[llc * 2 :, :] = data_tiles[5, :]

        f3 = data_tiles[6, :]

        f4[:, llc * 0 : llc * 1] = data_tiles[7, :]
        f4[:, llc * 1 : llc * 2] = data_tiles[8, :]
        f4[:, llc * 2 :] = data_tiles[9, :]

        f5[:, llc * 0 : llc * 1] = data_tiles[10, :]
        f5[:, llc * 1 : llc * 2] = data_tiles[11, :]
        f5[:, llc * 2 :] = data_tiles[12, :]

    # 3D slices on 13 tiles
    elif num_dims == 4:
        for k in range(nk):
            f1[k, llc * 0 : llc * 1, :] = data_tiles[k, 0, :]

            f1[k, llc * 1 : llc * 2, :] = data_tiles[k, 1, :]
            f1[k, llc * 2 :, :] = data_tiles[k, 2, :]

            f2[k, llc * 0 : llc * 1, :] = data_tiles[k, 3, :]
            f2[k, llc * 1 : llc * 2, :] = data_tiles[k, 4, :]
            f2[k, llc * 2 :, :] = data_tiles[k, 5, :]

            f3[k, :] = data_tiles[k, 6, :]

            f4[k, :, llc * 0 : llc * 1] = data_tiles[k, 7, :]
            f4[k, :, llc * 1 : llc * 2] = data_tiles[k, 8, :]
            f4[k, :, llc * 2 :] = data_tiles[k, 9, :]

            f5[k, :, llc * 0 : llc * 1] = data_tiles[k, 10, :]
            f5[k, :, llc * 1 : llc * 2] = data_tiles[k, 11, :]
            f5[k, :, llc * 2 :] = data_tiles[k, 12, :]

    # 4D slices on 13 tiles
    elif num_dims == 5:
        for l in range(nl):
            for k in range(nk):
                f1[l, k, llc * 0 : llc * 1, :] = data_tiles[l, k, 0, :]

                f1[l, k, llc * 1 : llc * 2, :] = data_tiles[l, k, 1, :]
                f1[l, k, llc * 2 :, :] = data_tiles[l, k, 2, :]

                f2[l, k, llc * 0 : llc * 1, :] = data_tiles[l, k, 3, :]
                f2[l, k, llc * 1 : llc * 2, :] = data_tiles[l, k, 4, :]
                f2[l, k, llc * 2 :, :] = data_tiles[l, k, 5, :]

                f3[l, k, :] = data_tiles[l, k, 6, :]

                f4[l, k, :, llc * 0 : llc * 1] = data_tiles[l, k, 7, :]
                f4[l, k, :, llc * 1 : llc * 2] = data_tiles[l, k, 8, :]
                f4[l, k, :, llc * 2 :] = data_tiles[l, k, 9, :]

                f5[l, k, :, llc * 0 : llc * 1] = data_tiles[l, k, 10, :]
                f5[l, k, :, llc * 1 : llc * 2] = data_tiles[l, k, 11, :]
                f5[l, k, :, llc * 2 :] = data_tiles[l, k, 12, :]

    # Build the F dictionary
    F = {}
    F[1] = f1
    F[2] = f2
    F[3] = f3
    F[4] = f4
    F[5] = f5

    return F


def llc_faces_to_compact(F, less_output=True):
    """

    Converts a dictionary containing five 'faces' of the lat-lon-cap grid
    and rearranges it to the 'compact' llc format.


    Parameters
    ----------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_compact : ndarray
        an array of dimension nl x nk x nj x ni

        F is in the llc compact format.

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_compact

    """

    # pull the individual faces out of the F dictionary
    f1 = F[1]
    f2 = F[2]
    f3 = F[3]
    f4 = F[4]
    f5 = F[5]

    # ascertain how many dimensions are in the faces (minimum 2, maximum 4)
    dims = f3.shape
    num_dims = len(dims)

    # data type of original faces
    arr_dtype = f1.dtype

    # the final dimension is always the llc #
    llc = dims[-1]

    # initialize the 'data_compact' array
    if num_dims == 2:  # we have a 2D slice (x,y)
        data_compact = np.zeros((13 * llc, llc), dtype=arr_dtype)

    elif num_dims == 3:  # 3D slice (x, y, time or depth)
        nk = dims[0]
        data_compact = np.zeros((nk, 13 * llc, llc), dtype=arr_dtype)

    elif num_dims == 4:  # 4D slice (x,y,time and depth)
        nl = dims[0]
        nk = dims[1]
        data_compact = np.zeros((nl, nk, 13 * llc, llc), dtype=arr_dtype)
    else:
        print("llc_faces_to_compact: can only handle faces that have 2, 3, or 4 dimensions!")
        return []

    if not less_output:
        print("llc_faces_to_compact: face 3 shape", f3.shape)

    if num_dims == 2:
        data_compact[: 3 * llc, :] = f1
        data_compact[3 * llc : 6 * llc, :] = f2
        data_compact[6 * llc : 7 * llc, :] = f3

        for f in range(8, 11):
            i1 = np.arange(0, llc) + (f - 8) * llc
            i2 = np.arange(0, 3 * llc, 3) + 7 * llc + f - 8
            data_compact[i2, :] = f4[:, i1]

        for f in range(11, 14):
            i1 = np.arange(0, llc) + (f - 11) * llc
            i2 = np.arange(0, 3 * llc, 3) + 10 * llc + f - 11
            data_compact[i2, :] = f5[:, i1]

    elif num_dims == 3:
        # loop through k indicies
        print("llc_faces_to_compact: data_compact array shape", data_compact.shape)

        for k in range(nk):
            data_compact[k, : 3 * llc, :] = f1[k, :]
            data_compact[k, 3 * llc : 6 * llc, :] = f2[k, :]
            data_compact[k, 6 * llc : 7 * llc, :] = f3[k, :]

            # if someone could explain why I have to transpose
            # f4 and f5 when num_dims =3 or 4 I would be so grateful.
            # Just could not figure this out.  Transposing works but why!?
            for f in range(8, 11):
                i1 = np.arange(0, llc) + (f - 8) * llc
                i2 = np.arange(0, 3 * llc, 3) + 7 * llc + f - 8

                data_compact[k, i2, :] = f4[k, 0:llc, i1].T

            for f in range(11, 14):
                i1 = np.arange(0, llc) + (f - 11) * llc
                i2 = np.arange(0, 3 * llc, 3) + 10 * llc + f - 11
                data_compact[k, i2, :] = f5[k, :, i1].T

    elif num_dims == 4:
        # loop through l and k indices
        for l in range(nl):
            for k in range(nk):
                data_compact[l, k, : 3 * llc, :] = f1[l, k, :]
                data_compact[l, k, 3 * llc : 6 * llc, :] = f2[l, k, :]
                data_compact[l, k, 6 * llc : 7 * llc, :] = f3[l, k, :]

                for f in range(8, 11):
                    i1 = np.arange(0, llc) + (f - 8) * llc
                    i2 = np.arange(0, 3 * llc, 3) + 7 * llc + f - 8
                    data_compact[l, k, i2, :] = f4[l, k, :, i1].T

                for f in range(11, 14):
                    i1 = np.arange(0, llc) + (f - 11) * llc
                    i2 = np.arange(0, 3 * llc, 3) + 10 * llc + f - 11
                    data_compact[l, k, i2, :] = f5[l, k, :, i1].T

    if not less_output:
        print("llc_faces_to_compact: data_compact array shape", data_compact.shape)
        print("llc_faces_to_compact: data_compact array dtype", data_compact.dtype)

    return data_compact


def llc_tiles_to_xda(data_tiles, var_type=None, grid_da=None, less_output=True, dim4=None, dim5=None):
    """
    Convert numpy or dask array in tiled format to xarray DataArray
    with minimal coordinates: (time,k,tile,j,i) ; (time,k,tile,j_g,i) etc...
    unless a DataArray or Dataset is provided as a template
    to provide more coordinate info

    4D field (5D array with tile dimension) Example:
    A 4D field (3D in space and 1D in time) living on tracer points with
    dimension order resulting from read_bin_llc.read_llc_to_tiles:

       >> array.shape
       [N_tiles, N_recs, N_z, N_y, N_x]

    We would read this in as follows:

        >> xda = llc_tiles_to_xda(data_tiles=array, var_type='c',
                                  dim4='depth', dim5='time')

    or equivalently

        >> xda = llc_tiles_to_xda(data_tiles=array, var_type='c',
                                  dim4='k', dim5='time')

    since 'depth' has been coded to revert to vertical coordinate 'k'...

    Note:
    1. for the 3D case, dim5 is not necessary
    2. for the 2D case, dim4 and dim5 are not necessary

    Special case!
    data_tiles can also be a 1D array ONLY if the user provides
    grid_da as a template for how to shape it back to a numpy array, then
    to DataArray.
    See calc_section_trsp._rotate_the_grid for an example usage.

    Parameters
    ----------
    data_tiles : numpy or dask+numpy array
        see above for specified dimension order

    var_type : string, optional
        Note: only optional if grid_da is provided!
        specification for where on the grid the variable lives
        'c' - grid cell center, i.e. tracer point, e.g. XC, THETA, ...
        'w' - west grid cell edge, e.g. dxG, zonal velocity, ...
        's' - south grid cell edge, e.g. dyG, meridional velocity, ...
        'z' - southwest grid cell edge, zeta/vorticity point, e.g. rAz

    grid_da : xarray DataArray, optional
        a DataArray or Dataset with the grid coordinates already loaded

    less_output : boolean, optional
        A debugging flag.  False = less debugging output

    dim4, dim5 : string, optional
        Specify name of fourth and fifth dimension, e.g. 'depth', 'k', or 'time'

    Returns
    -------
    da : xarray DataArray
    """

    if var_type is None and grid_da is None:
        raise TypeError('Must specify var_type="c","w","s", or "z" if grid_da is not provided')

    # Test for special case: 1D data
    if len(data_tiles.shape) == 1:
        if grid_da is None:
            raise TypeError("If converting 1D array, must specify grid_da as template")

        if not less_output:
            print("Found 1D array, will use grid_da input to shape it")
    elif len(data_tiles.shape) > 5:
        raise TypeError("Found unfamiliar array shape: ", data_tiles.shape)

    # If a DataArray or Dataset is given to model after, use this first!
    if grid_da is not None:
        # Add case for 1D array
        # This is like the gcmfaces routine convert2gcmfaces or convert2array
        # except it's practically two lines of code
        if len(data_tiles.shape) == 1:
            data_tiles = np.reshape(data_tiles, np.shape(grid_da.values))

        # don't copy over attributes from grid_da.  Let user specify own attributes
        da = xr.DataArray(data=data_tiles, coords=grid_da.coords.variables, dims=grid_da.dims, attrs=dict())
        return da

    # Provide dims and coords based on grid location
    if var_type == "c":
        da = _make_data_array(data_tiles, "i", "j", "k", less_output, dim4, dim5)

    elif var_type == "w":
        da = _make_data_array(data_tiles, "i_g", "j", "k", less_output, dim4, dim5)

    elif var_type == "s":
        da = _make_data_array(data_tiles, "i", "j_g", "k", less_output, dim4, dim5)

    elif var_type == "z":
        da = _make_data_array(data_tiles, "i_g", "j_g", "k", less_output, dim4, dim5)

    else:
        raise NotImplementedError("Can only take 'c', 'w', 's', or 'z', other types not implemented.")

    return da


def _make_data_array(data_tiles, iVar, jVar, kVar, less_output=False, dim4=None, dim5=None):
    """Non user facing function to make a data array from tiled numpy/dask array
    and strings denoting grid location

    Note that here, I'm including the "tiles" dimension...
    so dim4 refers to index vector d_4, and dim5 refers to index d_5
    No user should have to deal with this though

    Parameters
    ----------
    data_tiles : numpy/dask array
        Probably loaded from binary via mds_io.read_bin_to_tiles and rearranged
        in llc_tiles_to_xda
    iVar : string
        denote x grid location, 'i' or 'i_g'
    jVar : string
        denote y grid location, 'j' or 'j_g'
    kVar : string
        denote x grid location, 'k' only implemented for now.
        possible to implement 'k_u' for e.g. vertical velocity ... at some point
    less_output : boolean, optional
        debugging flag, False => print more
    dim4, dim5 : string, optional
        Specify name of fourth and fifth dimension, e.g. 'depth', 'k', or 'time'

    Returns
    -------
    da : xarray DataArray
    """

    # Save shape and num dimensions for below
    data_shape = data_tiles.shape
    Ndims = len(data_shape)

    # Create minimal coordinate information
    i = np.arange(data_shape[-1])
    j = np.arange(data_shape[-2])
    tiles = np.arange(data_shape[-3])
    d_4 = []
    d_5 = []
    if len(data_shape) > 3:
        if dim4 is None:
            raise TypeError("Please specify 4th dimension as dim4='depth' or dim4='time'")
        d_4 = np.arange(data_shape[-4])

    if len(data_shape) > 4:
        if dim5 is None:
            raise TypeError("Please specify 5th dimension as dim5='depth' or dim5='time'")
        d_5 = np.arange(data_shape[-5])

    # Create dims tuple, which will at least have
    # e.g. ('tile','j','i') for a 'c' variable
    dims = ("tile", jVar, iVar)

    # Coordinates will be a dictionary of 1 dimensional xarray DataArrays
    # each with their own set of attributes
    coords = OrderedDict()
    if Ndims > 3:
        if dim4 == "depth":
            mydim = kVar
        else:
            mydim = dim4

        dims = (mydim,) + dims
        attrs = dimensions[mydim] if mydim in dimensions else {}
        xda4 = xr.DataArray(data=d_4, coords={mydim: d_4}, dims=(mydim,), attrs=attrs)
        coords[mydim] = xda4

    if Ndims > 4:
        if dim5 == "depth":
            mydim = kVar
        else:
            mydim = dim5

        dims = (mydim,) + dims
        attrs = dimensions[mydim] if mydim in dimensions else {}
        xda5 = xr.DataArray(data=d_5, coords={mydim: d_5}, dims=(mydim,), attrs=attrs)
        coords[mydim] = xda5

    # Now add the standard coordinates
    tile_da = xr.DataArray(
        data=tiles, coords={"tile": tiles}, dims=("tile",), attrs=OrderedDict([("standard_name", "tile_index")])
    )
    j_da = xr.DataArray(data=j, coords={jVar: j}, dims=(jVar,), attrs=dimensions[jVar]["attrs"])
    i_da = xr.DataArray(data=i, coords={iVar: i}, dims=(iVar,), attrs=dimensions[iVar]["attrs"])

    coords["tile"] = tile_da
    coords[jVar] = j_da
    coords[iVar] = i_da

    return xr.DataArray(data=data_tiles, coords=coords, dims=dims)


def nat2globe(llc):
    """
    Reorder a native 1170-by-90 format array (llc) to a geographically
    contiguous global 360-by-360 array (glb) for visualization.

    Parameters:
    llc (np.ndarray): Input array of shape (1170, 90).

    Returns:
    np.ndarray: Output array of shape (360, 360).

    """
    # Get the size of the input array
    nx = llc.shape[1]

    # Calculate extended dimensions
    nx2 = nx * 2
    nx3 = nx * 3
    nx4 = nx * 4

    # Initialize the global array
    glb = np.zeros((nx4, nx4), dtype=np.float32)

    # Face 1
    glb[0:nx3, 0:nx] = llc[0:nx3, :]

    # Face 2
    ioff = nx
    glb[0:nx3, nx:nx2] = llc[nx3 : nx3 * 2, :]

    # Face 3
    glb[nx3:, 0:nx] = np.rot90(llc[2 * nx3 : 2 * nx3 + nx, :], k=3)

    # Face 4
    dum = np.zeros((nx, nx3), dtype=np.float32)
    dum[:, :] = llc[2 * nx3 + nx : 3 * nx3 + nx, :].reshape(nx, nx3)
    glb[0:nx3, nx2:nx3] = np.rot90(dum, k=1)

    # Face 5
    dum[:, :] = llc[3 * nx3 + nx :, :].reshape(nx, nx3)
    glb[0:nx3, nx3:] = np.rot90(dum, k=1)

    return glb
