"""
Dose Comparison Functions

Functions for comparing dose distributions:
- 2D slice comparisons with heatmaps
- 1D line profile comparisons
- Depth-dose curve comparisons
- Statistical analysis
"""

from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

# Optional imports for plotting (may not be available in all environments)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from scipy import interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Line Profile Extraction
# =============================================================================

def extract_line_profile(
    img: sitk.Image,
    axis: str = "x",
    position: Optional[Tuple[Optional[float], Optional[float]]] = None,
    slice_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a 1D line profile from a 3D image.
    
    Parameters
    ----------
    img : sitk.Image
        Input image
    axis : str
        Profile direction: 'x' (horizontal), 'y' (vertical), 'z' (depth)
    position : tuple, optional
        (x_pos, y_pos) physical position in mm. Use None for center.
        - For X profile: y_pos is used (x_pos ignored)
        - For Y profile: x_pos is used (y_pos ignored)
        - For Z profile: both x_pos and y_pos are used
    slice_idx : int, optional
        Slice index for z-dimension. If None, uses middle slice.
        
    Returns
    -------
    tuple
        (positions_mm, doses) - position array and dose values
    """
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()  # (dx, dy, dz)
    origin = img.GetOrigin()
    
    # Default to center indices
    center_x = arr.shape[2] // 2
    center_y = arr.shape[1] // 2
    center_z = arr.shape[0] // 2
    
    # Override with position if specified
    if position is not None:
        if len(position) > 0 and position[0] is not None:
            center_x = int(round((position[0] - origin[0]) / spacing[0]))
        if len(position) > 1 and position[1] is not None:
            center_y = int(round((position[1] - origin[1]) / spacing[1]))
    
    if slice_idx is None:
        slice_idx = center_z
    
    if axis.lower() == 'x':
        profile = arr[slice_idx, center_y, :]
        positions = origin[0] + np.arange(len(profile)) * spacing[0]
    elif axis.lower() == 'y':
        profile = arr[slice_idx, :, center_x]
        positions = origin[1] + np.arange(len(profile)) * spacing[1]
    elif axis.lower() == 'z':
        profile = arr[:, center_y, center_x]
        positions = origin[2] + np.arange(len(profile)) * spacing[2]
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'x', 'y', or 'z'.")
    
    return positions, profile


# =============================================================================
# 2D Comparison Functions
# =============================================================================

def compare_2d_slices(
    img1: sitk.Image,
    img2: sitk.Image,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    names: Tuple[str, str] = ("Image 1", "Image 2"),
    show_diff: bool = True,
    colorscale: str = "Magma",
    diff_colorscale: str = "RdBu_r"
) -> "go.Figure":
    """
    Compare 2D slices from two dose images side by side.
    
    Parameters
    ----------
    img1, img2 : sitk.Image
        Input images (should have same geometry, use resample_to_reference first if not)
    slice_idx : int, optional
        Slice index. If None, uses middle slice.
    axis : int
        Axis for slicing (0=sagittal, 1=coronal, 2=axial)
    names : tuple
        Labels for the two images
    show_diff : bool
        If True, show a third panel with difference map
    colorscale : str
        Colorscale for dose images
    diff_colorscale : str
        Colorscale for difference map
        
    Returns
    -------
    go.Figure
        Plotly figure with comparison
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for compare_2d_slices. Install with: pip install plotly")
    
    arr1 = sitk.GetArrayFromImage(img1)  # (z, y, x)
    arr2 = sitk.GetArrayFromImage(img2)
    
    spacing = img1.GetSpacing()  # (dx, dy, dz)
    origin = img1.GetOrigin()    # (x0, y0, z0)
    
    # Select slice
    if slice_idx is None:
        slice_idx = arr1.shape[axis] // 2
    
    # Extract 2D slice and determine physical coordinates
    if axis == 0:  # axial slice
        slice1, slice2 = arr1[slice_idx, :, :], arr2[slice_idx, :, :]
        xlabel, ylabel = "X (mm)", "Y (mm)"
        sp_x, sp_y = spacing[0], spacing[1]
        orig_x, orig_y = origin[0], origin[1]
    elif axis == 1:  # coronal slice
        slice1, slice2 = arr1[:, slice_idx, :], arr2[:, slice_idx, :]
        xlabel, ylabel = "X (mm)", "Z (mm)"
        sp_x, sp_y = spacing[0], spacing[2]
        orig_x, orig_y = origin[0], origin[2]
    else:  # axis == 2, sagittal slice
        slice1, slice2 = arr1[:, :, slice_idx], arr2[:, :, slice_idx]
        xlabel, ylabel = "Y (mm)", "Z (mm)"
        sp_x, sp_y = spacing[1], spacing[2]
        orig_x, orig_y = origin[1], origin[2]
    
    vmax = max(slice1.max(), slice2.max())
    
    ncols = 3 if show_diff else 2
    titles = [names[0], names[1]]
    if show_diff:
        titles.append("Difference")
    
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles, horizontal_spacing=0.08)
    
    # Create position arrays with proper physical coordinates
    x = orig_x + np.arange(slice1.shape[1]) * sp_x
    y = orig_y + np.arange(slice1.shape[0]) * sp_y
    
    fig.add_trace(
        go.Heatmap(z=slice1, x=x, y=y, colorscale=colorscale, zmin=0, zmax=vmax, 
                   colorbar=dict(x=0.28, len=0.8) if show_diff else dict(x=0.45, len=0.8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=slice2, x=x, y=y, colorscale=colorscale, zmin=0, zmax=vmax,
                   colorbar=dict(x=0.63, len=0.8) if show_diff else dict(x=1.0, len=0.8)),
        row=1, col=2
    )
    
    if show_diff:
        diff = slice1 - slice2
        diff_max = max(abs(diff.min()), abs(diff.max()))
        fig.add_trace(
            go.Heatmap(z=diff, x=x, y=y, colorscale=diff_colorscale, 
                       zmin=-diff_max, zmax=diff_max, colorbar=dict(x=1.0, len=0.8)),
            row=1, col=3
        )
    
    for i in range(1, ncols + 1):
        fig.update_xaxes(title_text=xlabel, row=1, col=i)
        fig.update_yaxes(title_text=ylabel, row=1, col=i, scaleanchor=f"x{i}" if i > 1 else "x")
    
    fig.update_layout(height=450, width=400 * ncols, margin=dict(l=20, r=20, t=40, b=20))
    
    return fig


# =============================================================================
# 1D Comparison Functions
# =============================================================================

def compare_1d_profiles(
    img1: sitk.Image,
    img2: sitk.Image,
    axes: Tuple[str, ...] = ("x", "y"),
    slice_idx: Optional[int] = None,
    depth_mm: Optional[float] = None,
    x_pos_mm: Optional[float] = None,
    y_pos_mm: Optional[float] = None,
    names: Tuple[str, str] = ("Image 1", "Image 2"),
    show_diff: bool = True
) -> "go.Figure":
    """
    Compare 1D line profiles between two images.
    
    Parameters
    ----------
    img1, img2 : sitk.Image
        Input images (should have same geometry)
    axes : tuple
        Which profiles to plot: 'x', 'y', 'z'
    slice_idx : int, optional
        Slice index (z). If None, uses middle slice.
    depth_mm : float, optional
        Physical depth (z-position) in mm. Overrides slice_idx if provided.
    x_pos_mm : float, optional
        X position (mm) for extracting Y and Z profiles. If None, uses center.
    y_pos_mm : float, optional
        Y position (mm) for extracting X and Z profiles. If None, uses center.
    names : tuple
        Labels for the two images
    show_diff : bool
        If True, show difference in subplot below
        
    Returns
    -------
    go.Figure
        Plotly figure with line profiles
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for compare_1d_profiles. Install with: pip install plotly")
    if not HAS_SCIPY:
        raise ImportError("scipy is required for compare_1d_profiles. Install with: pip install scipy")
    
    naxes = len(axes)
    nrows = 2 if show_diff else 1
    
    origin = img1.GetOrigin()
    spacing = img1.GetSpacing()
    size = img1.GetSize()
    
    # Convert depth_mm to slice_idx if provided
    if depth_mm is not None:
        slice_idx = int(round((depth_mm - origin[2]) / spacing[2]))
        print(f"Depth z={depth_mm} mm -> slice index {slice_idx}")
    
    # Convert x_pos_mm and y_pos_mm to indices
    if x_pos_mm is not None:
        x_idx = int(round((x_pos_mm - origin[0]) / spacing[0]))
        print(f"X position {x_pos_mm} mm -> x index {x_idx}")
    else:
        x_idx = size[0] // 2
        x_pos_mm = origin[0] + x_idx * spacing[0]
    
    if y_pos_mm is not None:
        y_idx = int(round((y_pos_mm - origin[1]) / spacing[1]))
        print(f"Y position {y_pos_mm} mm -> y index {y_idx}")
    else:
        y_idx = size[1] // 2
        y_pos_mm = origin[1] + y_idx * spacing[1]
    
    titles = []
    for ax in axes:
        title = f"{ax.upper()}-Profile"
        annotations = []
        if ax.lower() == 'x':
            annotations.append(f"y={y_pos_mm:.1f}")
        elif ax.lower() == 'y':
            annotations.append(f"x={x_pos_mm:.1f}")
        elif ax.lower() == 'z':
            annotations.append(f"x={x_pos_mm:.1f}, y={y_pos_mm:.1f}")
        if depth_mm is not None and ax.lower() != 'z':
            annotations.append(f"z={depth_mm:.1f}")
        if annotations:
            title += f" @ {', '.join(annotations)} mm"
        titles.append(title)
    if show_diff:
        for ax in axes:
            titles.append(f"{ax.upper()}-Difference")
    
    fig = make_subplots(
        rows=nrows, cols=naxes,
        subplot_titles=titles,
        vertical_spacing=0.15
    )
    
    colors = ['#636EFA', '#EF553B']  # Blue and red
    
    for i, ax in enumerate(axes):
        # Set position based on axis
        if ax.lower() == 'x':
            position = (None, y_pos_mm)
        elif ax.lower() == 'y':
            position = (x_pos_mm, None)
        else:  # z
            position = (x_pos_mm, y_pos_mm)
        
        pos1, prof1 = extract_line_profile(img1, axis=ax, position=position, slice_idx=slice_idx)
        pos2, prof2 = extract_line_profile(img2, axis=ax, position=position, slice_idx=slice_idx)
        
        fig.add_trace(
            go.Scatter(x=pos1, y=prof1, mode='lines', name=names[0], 
                       line=dict(color=colors[0], width=2),
                       legendgroup='img1', showlegend=(i == 0)),
            row=1, col=i + 1
        )
        fig.add_trace(
            go.Scatter(x=pos2, y=prof2, mode='lines', name=names[1],
                       line=dict(color=colors[1], width=2, dash='solid'),
                       legendgroup='img2', showlegend=(i == 0)),
            row=1, col=i + 1
        )
        
        # Interpolate to common positions for difference
        if show_diff:
            f2 = interpolate.interp1d(pos2, prof2, kind='linear', fill_value=0, bounds_error=False)
            diff = prof1 - f2(pos1)
            
            fig.add_trace(
                go.Scatter(x=pos1, y=diff, mode='lines', name='Difference',
                           line=dict(color='#00CC96', width=2),
                           legendgroup='diff', showlegend=(i == 0)),
                row=2, col=i + 1
            )
            fig.update_xaxes(title_text="Position (mm)", row=2, col=i + 1)
            fig.update_yaxes(title_text="Dose Diff (Gy)", row=2, col=i + 1)
        
        fig.update_xaxes(title_text="Position (mm)", row=1, col=i + 1)
        fig.update_yaxes(title_text="Dose (Gy)", row=1, col=i + 1)
    
    fig.update_layout(
        height=300 * nrows + 100,
        width=450 * naxes,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=20, r=20, t=40, b=60)
    )
    
    return fig


def compare_depth_dose(
    img1: sitk.Image,
    img2: sitk.Image,
    names: Tuple[str, str] = ("Image 1", "Image 2"),
    normalize: bool = False
) -> "go.Figure":
    """
    Compare depth-dose curves (along z-axis through center).
    
    Parameters
    ----------
    img1, img2 : sitk.Image
        Input images
    names : tuple
        Labels for the two images
    normalize : bool
        If True, normalize both curves to their maximum
        
    Returns
    -------
    go.Figure
        Plotly figure with depth-dose curves
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for compare_depth_dose. Install with: pip install plotly")
    
    pos1, prof1 = extract_line_profile(img1, axis='z')
    pos2, prof2 = extract_line_profile(img2, axis='z')
    
    if normalize:
        prof1 = prof1 / prof1.max() * 100
        prof2 = prof2 / prof2.max() * 100
        ylabel = "Relative Dose (%)"
    else:
        ylabel = "Dose (Gy)"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos1, y=prof1, mode='lines', name=names[0],
                             line=dict(width=2)))
    fig.add_trace(go.Scatter(x=pos2, y=prof2, mode='lines', name=names[1],
                             line=dict(width=2, dash='solid')))
    
    fig.update_layout(
        xaxis_title="Depth (mm)",
        yaxis_title=ylabel,
        title="Depth-Dose Curve Comparison",
        height=400,
        width=700
    )
    
    return fig


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_dose_difference_stats(
    img1: sitk.Image, 
    img2: sitk.Image, 
    dose_threshold: float = 0.1
) -> dict:
    """
    Compute statistics of dose differences.
    
    Parameters
    ----------
    img1, img2 : sitk.Image
        Input images (should have same geometry)
    dose_threshold : float
        Only consider voxels where dose > threshold * max_dose
        
    Returns
    -------
    dict
        Statistics including mean, std, max difference, etc.
    """
    arr1 = sitk.GetArrayFromImage(img1)
    arr2 = sitk.GetArrayFromImage(img2)
    
    max_dose = max(arr1.max(), arr2.max())
    mask = (arr1 > dose_threshold * max_dose) | (arr2 > dose_threshold * max_dose)
    
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    
    # Relative difference (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(arr1 > 0.01 * max_dose, diff / arr1 * 100, 0)
    
    return {
        "mean_diff": float(np.mean(diff[mask])),
        "std_diff": float(np.std(diff[mask])),
        "max_diff": float(np.max(abs_diff[mask])),
        "mean_abs_diff": float(np.mean(abs_diff[mask])),
        "mean_rel_diff_pct": float(np.mean(rel_diff[mask])),
        "max_rel_diff_pct": float(np.max(np.abs(rel_diff[mask]))),
        "max_dose_img1": float(arr1.max()),
        "max_dose_img2": float(arr2.max()),
    }
