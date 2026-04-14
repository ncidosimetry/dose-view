# Dose View

A Python package for dose distribution visualization and comparison.

## Features

- **I/O**: Read CT and dose images from various formats (DICOM RTDose, MetaImage, NIfTI, NRRD)
- **Resampling**: Align images with different geometries
- **Visualization**: Create triplanar overlays showing dose on CT
- **Comparison**: Compare dose distributions with 2D heatmaps, 1D profiles, and statistical analysis
- **Patient Data Management**: Organize DICOM studies and reference frames

## Installation

```bash
pip install -e .

# For full features (DICOM, interactive plots, interpolation)
pip install -e ".[full]"
```

## Usage

### Reading Dose Images

```python
from dose_view import read_dose_image, read_ct_any

# Read dose from various formats
dose = read_dose_image("dose.dcm")       # DICOM RTDose
dose = read_dose_image("dose.mha")       # MetaImage
dose = read_dose_image("dose.nii.gz")    # NIfTI

# Read CT
ct = read_ct_any("path/to/ct_directory")
```

### Comparing Dose Distributions

```python
from dose_view import (
    read_dose_image, 
    resample_to_reference,
    compare_2d_slices,
    compare_1d_profiles,
    compute_dose_difference_stats
)

# Read two dose distributions
tps_dose = read_dose_image("tps_dose.dcm")
mc_dose = read_dose_image("mc_dose.mha")

# Resample to common geometry
mc_resampled = resample_to_reference(mc_dose, tps_dose)

# 2D comparison with heatmaps
fig = compare_2d_slices(tps_dose, mc_resampled, names=("TPS", "MC"))
fig.show()

# 1D profile comparison
fig = compare_1d_profiles(tps_dose, mc_resampled, axes=("x", "y", "z"))
fig.show()

# Statistical analysis
stats = compute_dose_difference_stats(tps_dose, mc_resampled)
print(f"Mean difference: {stats['mean_diff']:.3f} Gy")
print(f"Max difference: {stats['max_diff']:.3f} Gy")
```

### Creating Triplanar Overlays

```python
from dose_view import save_triplanar_overlays

save_triplanar_overlays(
    ct_path="path/to/ct_directory",
    rtdose_path="path/to/dose.dcm",
    title="Patient Dose Distribution",
    output=True,
    output_dir="./figures"
)
```

## Package Structure

```
dose_view/
├── __init__.py           # Main package exports
├── io.py                 # Image I/O, resampling, and utilities
├── comparison.py         # Dose comparison functions
├── patient.py            # Patient data management
├── visualization.py      # Triplanar visualization
└── cli.py               # Command-line interface
```

## Modules

### `dose_view.io`
- `read_dose_image()` - Read dose from various formats (DICOM, MetaImage, NIfTI, etc.)
- `read_ct_any()` - Read CT from DICOM series or single file
- `read_rtdose_file()` - Read DICOM RTDose with proper scaling
- `find_rtdose_in_dir()` - Find RTDOSE file in directory
- `get_dose()` - Get dose from file or directory
- `resample_like()` - Simple resampling to reference geometry
- `resample_to_reference()` - Resample with interpolator options
- `resample_to_common_grid()` - Resample two images to common grid
- `resample_to_spacing()` - Resample to specific spacing
- `window_ct()` - Apply CT windowing
- `get_image_info()` / `print_image_info()` - Image information
- `create_empty_image()` - Create empty image with geometry
- `physical_to_index()` / `index_to_physical()` - Coordinate conversion

### `dose_view.comparison`
- `extract_line_profile()` - Extract 1D profile from image
- `compare_2d_slices()` - Compare 2D slices with heatmaps
- `compare_1d_profiles()` - Compare 1D line profiles
- `compare_depth_dose()` - Compare depth-dose curves
- `compute_dose_difference_stats()` - Statistical analysis of differences

### `dose_view.patient`
- `get_patient_data()` - Retrieve patient data from JSON
- `ct_to_dose()` - Map CT to associated dose files
- `dose_to_ct()` - Map dose files to CT

### `dose_view.visualization`
- `extract_triplanar_arrays()` - Extract axial/coronal/sagittal slices
- `overlay_panel()` - Create single overlay panel
- `save_triplanar_overlays()` - Create and save triplanar figure

## Command Line Interface

List available functions:
```bash
dose-view list
```

Call a function:
```bash
dose-view io.read_dose_image --kwargs '{"filepath": "dose.dcm"}'
dose-view comparison.compute_dose_difference_stats --kwargs '{"img1": "...", "img2": "..."}'
```

## Dependencies

**Required:**
- numpy
- SimpleITK
- matplotlib

**Optional (for full features):**
- pydicom - DICOM RTDose reading with metadata
- plotly - Interactive comparison plots
- scipy - Interpolation for profile comparison

## Development

Run tests:
```bash
python tests/test_import.py
```
