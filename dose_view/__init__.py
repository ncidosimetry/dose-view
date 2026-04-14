"""
Dose View Package

A Python package for dose distribution visualization and comparison.
Includes tools for reading DICOM dose files, creating triplanar overlays,
comparing dose distributions, and managing patient data.

Example Usage
-------------
>>> from dose_view import read_dose_image, compare_2d_slices, resample_to_reference
>>> 
>>> # Read dose images
>>> tps_dose = read_dose_image("tps_dose.dcm")
>>> mc_dose = read_dose_image("mc_dose.mha")
>>> 
>>> # Resample to common geometry
>>> mc_resampled = resample_to_reference(mc_dose, tps_dose)
>>> 
>>> # Compare
>>> fig = compare_2d_slices(tps_dose, mc_resampled, names=("TPS", "MC"))
>>> fig.show()
"""

__version__ = "0.1.0"

# I/O, resampling, and utility functions
from .io import (
    # Basic I/O
    read_ct_any,
    read_rtdose_file,
    read_dose_image,
    find_rtdose_in_dir,
    get_dose,
    # Resampling
    resample_like,
    resample_to_reference,
    resample_to_common_grid,
    resample_to_spacing,
    # Utilities
    window_ct,
    get_image_info,
    print_image_info,
    create_empty_image,
    physical_to_index,
    index_to_physical,
)

# Patient data management
from .patient import get_patient_data, ct_to_dose, dose_to_ct

# Visualization functions
from .visualization import (
    extract_triplanar_arrays,
    overlay_panel,
    save_triplanar_overlays,
)

# Comparison functions
from .comparison import (
    extract_line_profile,
    compare_2d_slices,
    compare_1d_profiles,
    compare_depth_dose,
    compute_dose_difference_stats,
)

__all__ = [
    # I/O
    "read_ct_any",
    "read_rtdose_file",
    "read_dose_image",
    "find_rtdose_in_dir",
    "get_dose",
    # Resampling
    "resample_like",
    "resample_to_reference",
    "resample_to_common_grid",
    "resample_to_spacing",
    # Utils
    "window_ct",
    "get_image_info",
    "print_image_info",
    "create_empty_image",
    "physical_to_index",
    "index_to_physical",
    # Patient
    "get_patient_data",
    "ct_to_dose",
    "dose_to_ct",
    # Visualization
    "extract_triplanar_arrays",
    "overlay_panel",
    "save_triplanar_overlays",
    # Comparison
    "extract_line_profile",
    "compare_2d_slices",
    "compare_1d_profiles",
    "compare_depth_dose",
    "compute_dose_difference_stats",
]
