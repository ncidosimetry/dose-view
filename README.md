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

## Render an existing dose-preview directory

The single-row preview renderer reproduces the dark **axial, sagittal, coronal**
layout with patient-coordinate axes, orientation labels, linked crosshairs,
a `turbo` overlay, and a shared scale based on the entire RTDOSE maximum.
The existing `save_triplanar_overlays` multi-row API keeps its original behavior.

Install the DICOM dependencies and point the command at a case directory:

```bash
pip install -e '.[dicom]'
dose-view preview /path/to/dose_preview/patient_case

# Process all case subdirectories, preserving their relative paths:
dose-view preview /path/to/dose_preview --recursive --output-dir /path/to/new_plots
```

The directory must contain `*_geometry.json` or `*.geometry.json` reports.
Both the original reports with `sources.CT` / `sources.RTDOSE` and the new
reports with `ct.path` / `dose.path` are supported. A single report path also works.
**The original DICOM files must be accessible:** PNGs and geometry JSON do not
contain the CT or dose pixels. Relative source paths resolve beside the report.
If a volume moved, pass a prefix map:

```bash
dose-view preview /path/to/patient_case \
  --path-map '{"/old/dicom":"/new/dicom"}'
```

Output defaults to a `dose_view/` subdirectory and contains one PNG and a fresh
`.geometry.json` report per input. Existing outputs require `--overwrite`;
source reports are preserved. Recursive discovery excludes the output tree.
Saved crosshairs, CT window, dose threshold, crop, and style settings are reused
when recorded. Old reports that did not record crop/title settings need explicit
overrides to reproduce those details; the default shows the full CT extent.

```bash
dose-view preview /path/to/patient_case \
  --kwargs '{"crop_mm":{"x":[-100,100],"y":[-100,100]},"subtitle":"Selected plan dose"}'
```

Python usage:

```python
from dose_view import plot_dose_preview, save_triplanar_overlay

reports = plot_dose_preview("/path/to/dose_preview/patient_case")

# Or render directly from DICOM without an existing preview:
report = save_triplanar_overlay(
    ct_path="/path/to/CT_series",
    rtdose_path="/path/to/RTDOSE.dcm",
    output_path="/path/to/triplanar.png",
    # Optional: validate the plan/structure/CT reference chain.
    rtplan_path="/path/to/RTPLAN.dcm",
    rtstruct_path="/path/to/RTSTRUCT.dcm",
)
```

`plot_dose_preview` accepts `output_dir`, `recursive`, `overwrite`, `path_map`,
and the renderer's display options: `pick` (`dose_center`, `maxdose`, `center`),
`center_xyz_mm`, `crop_mm`, `ct_wl`, `ct_ww`, `dose_threshold_fraction`,
`center_threshold_fraction`, `alpha`, `cmap`, `title`, `subtitle`, `dpi`, `workers`.
Supplying `pick` or `center_threshold_fraction` recomputes the center unless
`center_xyz_mm` is explicitly supplied. Coordinates are DICOM patient LPS mm.
The default center is weighted by dose at or above 50% of the maximum; the
default overlay hides values below 10% of the maximum.

Each render rereads DICOM and checks CT slice geometry, patient/study/frame
agreement, dose scaling, and frame offsets. Supplied plan/structure paths are
also checked. Missing optional references remain unchecked in the report.
Historical checks in input JSON are not reused as validation. Different grids,
oblique directions, and nonuniform/descending dose offsets are sampled in
patient coordinates; points outside a grid are masked. Offset interpretation
follows [DICOM PS3.3 C.8.8.3.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.3.2.html).
No registration or dose renormalization is applied; `RELATIVE` units and
`EFFECTIVE` dose type retain their meaning. Plotting does not establish
full-course dose coverage or a validated link to a clinical registry.

## Package Structure

```
dose_view/
├── __init__.py           # Main package exports
├── io.py                 # Image I/O, resampling, and utilities
├── comparison.py         # Dose comparison functions
├── patient.py            # Patient data management
├── visualization.py      # Triplanar visualization
├── triplanar.py          # Patient-coordinate preview renderer
├── preview.py            # Geometry sidecar / directory input
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

Patient lookup uses exact identifiers. If an inventory uses aliases, supply an
explicit mapping; the library does not infer study prefixes or site suffixes:

```python
from dose_view import get_patient_data

entry = get_patient_data(
    "example-alias",
    inventory,
    aliases={"example-alias": "example-record"},
)
```

Existing callers that relied on automatic prefix/suffix guessing must supply
their own alias mapping.

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
python -m unittest discover -s tests -v
```
