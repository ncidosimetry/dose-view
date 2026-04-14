"""
Dose Image I/O and Processing Functions

Read and write dose images from various formats:
- DICOM RTDose
- MetaImage (.mha/.mhd)
- NIfTI (.nii/.nii.gz)
- NRRD

Also includes resampling and image processing utilities.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

# Optional import for pydicom (not needed for all functions)
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def read_ct_any(path: Union[str, Path]) -> sitk.Image:
    """
    Read CT from a DICOM series folder or a single image file.
    
    Parameters
    ----------
    path : str or Path
        Path to DICOM series directory or single image file
        
    Returns
    -------
    sitk.Image
        CT image as float32
    """
    path = str(path)
    if os.path.isdir(path):
        sr = sitk.ImageSeriesReader()
        series = sr.GetGDCMSeriesIDs(path)
        if not series:
            raise RuntimeError(f"No DICOM series in {path}")
        files = sr.GetGDCMSeriesFileNames(path, series[0])
        sr.SetFileNames(files)
        img = sr.Execute()
    else:
        img = sitk.ReadImage(path)
    return sitk.Cast(img, sitk.sitkFloat32)


def read_rtdose_file(file_path: Union[str, Path]) -> sitk.Image:
    """
    Read DICOM RTDose file with proper scaling.
    
    Parameters
    ----------
    file_path : str or Path
        Path to RTDOSE file
        
    Returns
    -------
    sitk.Image
        Dose image in Gy
    """
    file_path = str(file_path)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No RTDOSE file found at {file_path}")
    
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(file_path)
        r.LoadPrivateTagsOn()
        r.ReadImageInformation()
        img = r.Execute()
        scale = float(r.GetMetaData("3004|000e")) if r.HasMetaDataKey("3004|000e") else 1.0
        dose = sitk.Cast(img, sitk.sitkFloat32) * scale
        dose.CopyInformation(img)
        return dose
    except Exception as e:
        raise RuntimeError(f"Failed to read RTDOSE file {file_path}: {str(e)}")


def resample_like(ref: sitk.Image, moving: sitk.Image, is_label: bool = False) -> sitk.Image:
    """
    Resample moving image to match reference image geometry.
    
    Parameters
    ----------
    ref : sitk.Image
        Reference image defining target geometry
    moving : sitk.Image
        Image to resample
    is_label : bool
        If True, use nearest neighbor interpolation
        
    Returns
    -------
    sitk.Image
        Resampled image
    """
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(moving, ref, sitk.Transform(), interp, 0.0, sitk.sitkFloat32)


def window_ct(ct_arr: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """
    Apply CT windowing (window level and width).
    
    Parameters
    ----------
    ct_arr : np.ndarray
        CT image array
    wl : float
        Window level (center)
    ww : float
        Window width
        
    Returns
    -------
    np.ndarray
        Windowed CT array normalized to [0, 1]
    """
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    x = np.clip(ct_arr, lo, hi)
    return (x - lo) / max(1e-6, (hi - lo))


# =============================================================================
# Multi-format Dose Reading
# =============================================================================

def read_dose_image(filepath: Union[str, Path]) -> sitk.Image:
    """
    Read a dose image from various formats using SimpleITK.
    Supports: DICOM RTDose, MetaImage (.mha/.mhd), NIfTI (.nii/.nii.gz), NRRD, etc.
    
    For DICOM RTDose, applies DoseGridScaling automatically.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the dose file
        
    Returns
    -------
    sitk.Image
        The dose image with proper scaling applied
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check if it's a DICOM file (RTDose)
    if filepath.suffix.lower() == '.dcm' or _is_dicom(filepath):
        return _read_dicom_rtdose(filepath)
    else:
        # Use SimpleITK for other formats (mha, mhd, nii, nii.gz, nrrd, etc.)
        return sitk.ReadImage(str(filepath), sitk.sitkFloat32)


def _is_dicom(filepath: Path) -> bool:
    """Check if file is a DICOM file by reading magic bytes."""
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False


def _read_dicom_rtdose(filepath: Path) -> sitk.Image:
    """
    Read DICOM RTDose file using pydicom for metadata and convert to SimpleITK image.
    Applies DoseGridScaling to get dose in Gy.
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for reading DICOM RTDose. Install with: pip install pydicom")
    
    ds = pydicom.dcmread(str(filepath))
    
    if getattr(ds, "Modality", None) != "RTDOSE":
        print(f"Warning: File modality is {getattr(ds, 'Modality', 'unknown')}, not RTDOSE")
    
    # Get pixel array and apply scaling
    dose_array = ds.pixel_array.astype(np.float32)
    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    dose_array *= scaling
    
    # Get spatial information
    ipp = [float(x) for x in ds.ImagePositionPatient]  # Origin (x, y, z)
    pixel_spacing = [float(x) for x in ds.PixelSpacing]  # (row_spacing, col_spacing)
    
    # Get slice thickness or GridFrameOffsetVector
    if hasattr(ds, 'GridFrameOffsetVector') and len(ds.GridFrameOffsetVector) > 1:
        slice_spacing = abs(float(ds.GridFrameOffsetVector[1]) - float(ds.GridFrameOffsetVector[0]))
    elif hasattr(ds, 'SliceThickness'):
        slice_spacing = float(ds.SliceThickness)
    else:
        slice_spacing = 1.0
    
    # Create SimpleITK image
    img = sitk.GetImageFromArray(dose_array)
    img.SetSpacing([pixel_spacing[1], pixel_spacing[0], slice_spacing])
    img.SetOrigin(ipp)
    img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    
    return img


def find_rtdose_in_dir(dicom_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find RTDOSE file in a DICOM directory.
    
    Parameters
    ----------
    dicom_dir : str or Path
        Path to directory containing DICOM files
        
    Returns
    -------
    Path or None
        Path to RTDOSE file if found, None otherwise
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for find_rtdose_in_dir. Install with: pip install pydicom")
    
    for dcm_path in sorted(Path(dicom_dir).glob("*.dcm")):
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
            if getattr(ds, "Modality", None) == "RTDOSE":
                return dcm_path
        except Exception:
            continue
    return None


def get_dose(dir_or_filepath: Union[str, Path]) -> sitk.Image:
    """
    Get dose image from a file path or directory.
    
    If given a directory, searches for RTDOSE file within it.
    
    Parameters
    ----------
    dir_or_filepath : str or Path
        Path to dose file or directory containing DICOM files
        
    Returns
    -------
    sitk.Image
        Dose image
    """
    dir_or_filepath = Path(dir_or_filepath)
    
    if dir_or_filepath.is_file():
        dose_path = dir_or_filepath
    else:
        dose_path = find_rtdose_in_dir(dir_or_filepath)
        if dose_path is None:
            raise FileNotFoundError(f"No RTDOSE found in {dir_or_filepath}")
    
    return read_dose_image(dose_path)


# =============================================================================
# Advanced Resampling Functions
# =============================================================================

def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0
) -> sitk.Image:
    """
    Resample moving image to match the geometry of reference image.
    
    Parameters
    ----------
    moving : sitk.Image
        Image to resample
    reference : sitk.Image
        Reference image defining the target geometry
    interpolator : int
        SimpleITK interpolator (sitkLinear, sitkNearestNeighbor, sitkBSpline)
    default_value : float
        Value for pixels outside the moving image
        
    Returns
    -------
    sitk.Image
        Resampled image with same geometry as reference
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())  # Identity transform
    
    return resampler.Execute(moving)


def resample_to_common_grid(
    img1: sitk.Image,
    img2: sitk.Image,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    interpolator: int = sitk.sitkLinear
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Resample two images to a common grid covering both.
    
    Parameters
    ----------
    img1, img2 : sitk.Image
        Input images
    target_spacing : tuple, optional
        Target spacing (dx, dy, dz). If None, uses finer spacing from inputs.
    interpolator : int
        SimpleITK interpolator
        
    Returns
    -------
    tuple
        (resampled_img1, resampled_img2) on common grid
    """
    # Determine target spacing
    if target_spacing is None:
        sp1 = np.array(img1.GetSpacing())
        sp2 = np.array(img2.GetSpacing())
        target_spacing = tuple(np.minimum(sp1, sp2))
    
    # Compute bounding box covering both images
    def get_physical_bounds(img):
        size = np.array(img.GetSize())
        origin = np.array(img.GetOrigin())
        spacing = np.array(img.GetSpacing())
        end = origin + size * spacing
        return origin, end
    
    orig1, end1 = get_physical_bounds(img1)
    orig2, end2 = get_physical_bounds(img2)
    
    common_origin = np.minimum(orig1, orig2)
    common_end = np.maximum(end1, end2)
    
    # Compute new size
    target_spacing = np.array(target_spacing)
    new_size = np.ceil((common_end - common_origin) / target_spacing).astype(int)
    
    # Create reference image for resampling
    ref_img = sitk.Image(new_size.tolist(), sitk.sitkFloat32)
    ref_img.SetSpacing(target_spacing.tolist())
    ref_img.SetOrigin(common_origin.tolist())
    ref_img.SetDirection(img1.GetDirection())
    
    # Resample both images
    resampled1 = resample_to_reference(img1, ref_img, interpolator)
    resampled2 = resample_to_reference(img2, ref_img, interpolator)
    
    return resampled1, resampled2


def resample_to_spacing(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0
) -> sitk.Image:
    """
    Resample image to a new spacing while preserving physical extent.
    
    Parameters
    ----------
    img : sitk.Image
        Image to resample
    target_spacing : tuple
        Target spacing (dx, dy, dz) in mm
    interpolator : int
        SimpleITK interpolator
    default_value : float
        Value for pixels outside the image
        
    Returns
    -------
    sitk.Image
        Resampled image with new spacing
    """
    original_spacing = np.array(img.GetSpacing())
    original_size = np.array(img.GetSize())
    target_spacing = np.array(target_spacing)
    
    # Calculate new size to maintain physical extent
    new_size = np.round(original_size * original_spacing / target_spacing).astype(int)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    
    return resampler.Execute(img)


# =============================================================================
# Image Information Utilities
# =============================================================================

def get_image_info(img: sitk.Image) -> dict:
    """
    Get basic information about a SimpleITK image.
    
    Parameters
    ----------
    img : sitk.Image
        Input image
        
    Returns
    -------
    dict
        Dictionary containing size, spacing, origin, direction, pixel_type, dimension
    """
    return {
        "size": img.GetSize(),           # (x, y, z)
        "spacing": img.GetSpacing(),     # (dx, dy, dz) in mm
        "origin": img.GetOrigin(),       # (x0, y0, z0) in mm
        "direction": img.GetDirection(),
        "pixel_type": img.GetPixelIDTypeAsString(),
        "dimension": img.GetDimension(),
    }


def print_image_info(img: sitk.Image, name: str = "Image") -> None:
    """
    Print image information to console.
    
    Parameters
    ----------
    img : sitk.Image
        Input image
    name : str
        Label for the image in output
    """
    info = get_image_info(img)
    print(f"{name}:")
    print(f"  Size (x,y,z): {info['size']}")
    print(f"  Spacing (mm): {info['spacing']}")
    print(f"  Origin (mm): {info['origin']}")
    print(f"  Pixel type: {info['pixel_type']}")


def create_empty_image(
    size: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Optional[Tuple[float, ...]] = None,
    pixel_type: int = sitk.sitkFloat32,
    fill_value: float = 0.0
) -> sitk.Image:
    """
    Create an empty SimpleITK image with specified geometry.
    
    Parameters
    ----------
    size : tuple of int
        Image size (nx, ny, nz) in voxels
    spacing : tuple of float
        Voxel spacing (dx, dy, dz) in mm
    origin : tuple of float
        Image origin (x0, y0, z0) in mm
    direction : tuple of float, optional
        Direction cosine matrix (9 elements). If None, uses identity.
    pixel_type : int
        SimpleITK pixel type (e.g., sitk.sitkFloat32, sitk.sitkUInt16)
    fill_value : float
        Value to fill the image with (default 0.0)
        
    Returns
    -------
    sitk.Image
        Empty image with specified geometry
    """
    img = sitk.Image([int(s) for s in size], pixel_type)
    img.SetSpacing([float(s) for s in spacing])
    img.SetOrigin([float(o) for o in origin])
    
    if direction is None:
        direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    img.SetDirection(direction)
    
    if fill_value != 0.0:
        arr = sitk.GetArrayFromImage(img)
        arr.fill(fill_value)
        filled = sitk.GetImageFromArray(arr)
        filled.CopyInformation(img)
        return filled
    
    return img


def physical_to_index(
    img: sitk.Image, 
    position: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """
    Convert physical position (mm) to voxel index.
    
    Parameters
    ----------
    img : sitk.Image
        The image
    position : tuple
        Physical position (x, y, z) in mm
        
    Returns
    -------
    tuple
        Voxel index (ix, iy, iz)
    """
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    pos = np.array(position)
    idx = np.round((pos - origin) / spacing).astype(int)
    return tuple(idx)


def index_to_physical(
    img: sitk.Image,
    index: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    """
    Convert voxel index to physical position (mm).
    
    Parameters
    ----------
    img : sitk.Image
        The image
    index : tuple
        Voxel index (ix, iy, iz)
        
    Returns
    -------
    tuple
        Physical position (x, y, z) in mm
    """
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    idx = np.array(index)
    pos = origin + idx * spacing
    return tuple(pos)

