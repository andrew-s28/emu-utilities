# emu-utilities

**ECCO Modeling Utilities (EMU) - Utilities**: *utilities for the utilities!*

This package provides functions for loading the outputs of the [ECCO Modeling Utilities (EMU)](https://ecco-group.org/docs/01_16_fukumori_emu_ecco_2024_03.pdf) into [xarray](https://docs.xarray.dev/en/stable/) data structures. This package includes support for outputs from the following EMU tools:

- **Adjoint Gradients**: Sensitivity of objective functions to control variables
- **Attribution**: Decomposed contributions of different forcing components
- **Convolution**: Time-lagged correlations between forcing and response
- **Forward Gradients**: Response of the model to perturbations
- **Sampling**: Time series extraction at specific locations

This package is *very* early in development, so use with caution and please reach out or submit an issue if you notice any problems.

## Usage

### Installation

Currently only cloning locally is supported:

```bash
git clone https://github.com/andrew-s28/emu-utilities.git
```

### Basic Import

```python
from emu_utilities import adjoint_gradient, attribution, convolution, forward_gradient, sampling, tracer
```

### Loading Adjoint Gradient Data

```python
# Load adjoint gradient data from EMU output folder
ds_adj = adjoint_gradient.load_adjoint_gradient("emu_adj_6_6_2_45_585_1")

# Print dataset to see available variables
print(ds_adj)
```

### Loading Attribution Data

```python
# Load attribution data from EMU output folder
ds_atr = attribution.load_attribution("emu_atrb_m_3_mask3d.-170.0_-120.0_-5.0_5.0_10.0_0.0_1")

# Plot time series comparing reference run and wind stress attribution run
plt.figure(figsize=(10, 6))
plt.plot(ds_atr.time, ds_atr.reference_run, label="Reference Run")
plt.plot(ds_atr.time, ds_atr.wind_stress, label="Wind Stress")
plt.legend()
plt.show()
```

### Loading Convolution Data

Convolution outputs both 1d (spatial sum) and 2d (maximum lag) datasets which have to be loaded in differently. Several helper methods can then be used to compute the explained variance from either the 1d or 2d datasets.

```python
# Load 1d convolution data from EMU output folder
ds_conv = convolution.load_1d_conv_gradient("emu_conv_6_6_2_45_585_1_26")

# Calculate and plot explained variance at a specific lag as a function of control variable
convolution.ctrl_variance(ds_conv, lag=10).plot.scatter()

# Calculate and plot explained variance of all controls at a specific lag
convolution.lagged_variance(ds_conv, variable="sum").plot.scatter()
```

```python
# Load 2d convolution data from EMU output folder
ds_conv = convolution.load_2d_conv_gradient("emu_conv_6_6_2_45_585_1_26")

# Calculate explained variance as a function of space
ds_sp_var = convolution.spatial_variance(ds, "sum")
```

### Loading Forward Gradient Data

```python
# Load forward gradient data from EMU output folder
ds_fgd = forward_gradient.load_forward_gradient("emu_fgrd_7_15_743_5_-1.00E-01")

# Print dataset to explore available variables
print(ds_fgd)
```

### Loading Sample Data

```python
# Load sample data from EMU output folder
ds_smp = sampling.load_sample("emu_samp_m_2_45_585_1")

# Plot time series of THETA
plt.figure(figsize=(10, 4))
plt.plot(ds_smp.time, ds_smp.THETA)
plt.title("THETA Time Series")
plt.xlabel("Time")
plt.ylabel("THETA")
plt.show()
```

### Loading Tracer Data

```python
# Load tracer data from EMU output folder
ds_trc = tracer.load_tracer_gradient("emu_trc_35_365_trc3d.-170.0_-120.0_-5.0_5.0_10.0_0.0")
```

### Resampling to Regular Lat-Lon Grid

EMU outputs are on the LLC-90 native ECCO grid. Any methods used to interpolate the central ECCO v4r4 state estimate fields to a regular lat-lon grid can be used here. A convinience method is included in this package, based on code from [ecco_v4_py](https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Interpolating_Fields_to_LatLon_Grid.html).

```python
from emu_utilities.resample import resample_ds

# Load adjoint gradient data
ds_adj = adjoint_gradient.load_adjoint_gradient("path/to/emu_adj_file")

# Define target grid parameters
new_grid_delta_lat = 1  # Grid spacing in latitude (degrees)
new_grid_delta_lon = 1  # Grid spacing in longitude (degrees)

new_grid_min_lat = -90
new_grid_max_lat = 90
new_grid_min_lon = -180
new_grid_max_lon = 180

# Auto-magically resamples every variable in the dataset with the correct coordinates!
ds_adj_resampled = resample_ds(
    ds_adj
    new_grid_min_lat,
    new_grid_max_lat,
    new_grid_delta_lat,
    new_grid_min_lon,
    new_grid_max_lon,
    new_grid_delta_lon,
    fill_value=np.nan,
    mapping_method="nearest_neighbor",
)

# Plot resampled data
plt.figure(figsize=(12, 6))
plt.pcolormesh(ds_adj_resampled["lon"], ds_adj_resampled["lat"], ds_adj_resampled["tauu"], cmap="viridis")
plt.colorbar(label="tauu")
plt.title("Resampled Zonal Wind Stress Sensitivity")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
```

### Visualizing Forward Gradient Data

```python
# Resample SSH data to lat-lon grid
lon_c, lat_c, _, _, ssh_latlon = resample_to_latlon(
    ds_fgd.xc,
    ds_fgd.yc,
    ds_fgd.ssh,
    new_grid_min_lat,
    new_grid_max_lat,
    new_grid_delta_lat,
    new_grid_min_lon,
    new_grid_max_lon,
    new_grid_delta_lon,
    fill_value=np.nan,
    mapping_method="nearest_neighbor",
)

# Plot SSH (multiplied by 1000 to convert to mm)
plt.figure(figsize=(12, 6))
plt.pcolormesh(lon_c[0], lat_c[:, 0], ssh_latlon[0]*1e3, shading='gouraud', cmap='RdBu_r')
plt.colorbar(label='SSH (mm)')
plt.title('Sea Surface Height')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
```

## Contributing

As this package is in early development, contributions are welcome! Here's how you can help:

1. Report bugs by [opening an issue](link-to-issues)
2. Suggest enhancements or new features
3. Submit pull requests with improvements

If you're interested in contributing, please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This package is licensed under the [MIT License](LICENSE).

This package uses code adapted from the [ecco_v4_py](https://ecco-v4-python-tutorial.readthedocs.io/) package (MIT License) within the `emu_utilities.resample` module; a copy of that license can be [found in that module file](emu_utilities/resample.py).
