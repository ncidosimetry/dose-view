"""
Visualization Functions

Functions for creating triplanar overlays and dose visualizations.
"""

from typing import Tuple, List, Optional

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .io import read_ct_any, read_rtdose_file, resample_like, window_ct


def extract_triplanar_arrays(
    ct_img: sitk.Image, 
    dose_img: sitk.Image, 
    pick: str = "maxdose"
) -> Tuple[Tuple[np.ndarray, np.ndarray, List[float]], ...]:
    """
    Extract triplanar (axial, coronal, sagittal) slices from CT and dose images.
    
    Parameters
    ----------
    ct_img : sitk.Image
        CT image
    dose_img : sitk.Image
        Dose image (must be aligned with CT)
    pick : str
        Slice selection method: "maxdose" or "center"
        
    Returns
    -------
    tuple
        ((ax_ct, ax_d, ext_ax), (co_ct, co_d, ext_co), (sa_ct, sa_d, ext_sa))
        Arrays and extents for axial, coronal, and sagittal views
    """
    ct = sitk.GetArrayFromImage(ct_img)     # (z,y,x)
    dose = sitk.GetArrayFromImage(dose_img) # (z,y,x)
    Z, Y, X = ct.shape
    
    if pick == "maxdose" and np.any(dose > 0):
        # Find max dose location
        max_dose_val = np.nanmax(dose)
        iz, iy, ix = np.unravel_index(np.nanargmax(dose), dose.shape)
        
        print(f"  Max dose value: {max_dose_val:.2f} Gy at position ({iz}, {iy}, {ix})")
        print(f"  Volume shape: {dose.shape}")
        print(f"  Dose range: {np.nanmin(dose[dose > 0]):.2f} - {np.nanmax(dose):.2f} Gy")
        
        # Check if max dose is at edge - if so, find a better representative slice
        edge_threshold = 5
        is_near_edge = (iz < edge_threshold or iz >= Z - edge_threshold or
                       iy < edge_threshold or iy >= Y - edge_threshold or
                       ix < edge_threshold or ix >= X - edge_threshold)
        
        if is_near_edge:
            print(f"  Max dose is near edge, finding better representative slice...")
            center_z_start, center_z_end = Z//4, 3*Z//4
            center_y_start, center_y_end = Y//4, 3*Y//4
            center_x_start, center_x_end = X//4, 3*X//4
            
            central_dose = dose[center_z_start:center_z_end, 
                              center_y_start:center_y_end, 
                              center_x_start:center_x_end]
            
            if np.any(central_dose > 0):
                rel_iz, rel_iy, rel_ix = np.unravel_index(np.nanargmax(central_dose), central_dose.shape)
                iz = rel_iz + center_z_start
                iy = rel_iy + center_y_start
                ix = rel_ix + center_x_start
                print(f"  Using central max dose at ({iz}, {iy}, {ix}) = {dose[iz, iy, ix]:.2f} Gy")
            else:
                print(f"  No dose in central region, using geometric center")
                iz, iy, ix = Z//2, Y//2, X//2
    else:
        iz, iy, ix = Z//2, Y//2, X//2
        print(f"  Using geometric center at ({iz}, {iy}, {ix})")

    # Extract slices
    ax_ct, ax_d = ct[iz, :, :], dose[iz, :, :]
    co_ct, co_d = ct[:, iy, :], dose[:, iy, :]
    sa_ct, sa_d = ct[:, :, ix], dose[:, :, ix]

    sx, sy, sz = ct_img.GetSpacing()
    ext_ax = [0, ax_ct.shape[1]*sx, 0, ax_ct.shape[0]*sy]
    ext_co = [0, co_ct.shape[1]*sx, 0, co_ct.shape[0]*sz]
    ext_sa = [0, sa_ct.shape[1]*sy, 0, sa_ct.shape[0]*sz]
    
    return (ax_ct, ax_d, ext_ax), (co_ct, co_d, ext_co), (sa_ct, sa_d, ext_sa)


def overlay_panel(
    ax: plt.Axes, 
    ct2d: np.ndarray, 
    d2d: np.ndarray, 
    extent: List[float],
    ct_wl: float = 40, 
    ct_ww: float = 400,
    dose_min: float = 0.0, 
    dose_max: Optional[float] = None, 
    alpha: float = 0.6, 
    thresh: float = 0.0,
    title: str = ""
) -> None:
    """
    Create an overlay panel showing CT with dose colormap.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to draw on
    ct2d : np.ndarray
        2D CT slice
    d2d : np.ndarray
        2D dose slice
    extent : list of float
        Image extent [xmin, xmax, ymin, ymax]
    ct_wl : float
        CT window level
    ct_ww : float
        CT window width
    dose_min : float
        Minimum dose for colormap
    dose_max : float, optional
        Maximum dose for colormap (uses 95th percentile if None)
    alpha : float
        Dose overlay transparency
    thresh : float
        Dose threshold for display (Gy)
    title : str
        Panel title
    """
    ct_gray = window_ct(ct2d, wl=ct_wl, ww=ct_ww)
    ax.imshow(ct_gray, cmap="gray", origin="lower", extent=extent)
    
    if dose_max is None:
        nz = d2d[d2d > 0]
        dose_max = float(np.percentile(nz, 95)) if nz.size else 1.0
    
    d_norm = np.clip((d2d - dose_min) / max(1e-6, (dose_max - dose_min)), 0, 1)
    
    if thresh > 0:
        alpha_mask = np.where(d2d >= thresh, alpha, 0.0)
        below_thresh_count = np.sum(d2d < thresh)
        print(f"    Pixels below {thresh} Gy threshold: {below_thresh_count}/{d2d.size}")
        
        from matplotlib import cm
        jet = cm.get_cmap('jet')
        dose_rgba = jet(d_norm)
        dose_rgba[:, :, 3] = alpha_mask
        ax.imshow(dose_rgba, origin="lower", extent=extent)
    else:
        ax.imshow(d_norm, cmap="jet", origin="lower", extent=extent, alpha=alpha)
    
    ax.set_title(title)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.grid(False)


def save_triplanar_overlays(
    ct_path: str, 
    rtdose_path: str, 
    title: str = "Dose over CT",
    pick: List[str] = ["maxdose", "center"], 
    ct_wl: float = 40, 
    ct_ww: float = 400,
    dose_min: float = 0.0, 
    dose_max: Optional[float] = None, 
    alpha: float = 0.6, 
    thresh: float = 0.1,
    output: bool = False, 
    output_dir: str = './'
) -> None:
    """
    Create and save triplanar dose overlays on CT.
    
    Parameters
    ----------
    ct_path : str
        Path to CT image
    rtdose_path : str
        Path to RTDOSE file
    title : str
        Figure title
    pick : list of str
        Slice selection methods for each row
    ct_wl : float
        CT window level
    ct_ww : float
        CT window width
    dose_min : float
        Minimum dose for colormap
    dose_max : float, optional
        Maximum dose for colormap
    alpha : float
        Dose overlay transparency
    thresh : float
        Dose threshold (Gy)
    output : bool
        If True, save to file instead of displaying
    output_dir : str
        Output directory for saved images
    """
    print(f"Processing: {title}")
    
    # Load & align
    ct = read_ct_any(ct_path)
    dose = read_rtdose_file(rtdose_path)
    dose_r = resample_like(ct, dose, is_label=False)

    # Create figure
    n_rows = len(pick)
    print(f"Number of rows: {n_rows}")
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows), constrained_layout=True)
    
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    # Determine dose_max
    (ax_ct, ax_d, ext_ax), _, _ = extract_triplanar_arrays(ct, dose_r, pick=pick[0])
    if dose_max is None:
        nz = ax_d[ax_d > 0]
        dose_max = float(np.percentile(nz, 95)) if nz.size else 1.0

    # Create colorbar
    norm = Normalize(vmin=dose_min, vmax=dose_max)
    sm = ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, shrink=0.8, aspect=20)
    cbar.set_label('Dose (Gy)', rotation=270, labelpad=15)

    # Plot each row
    for row_idx, pick_method in enumerate(pick):
        (ax_ct, ax_d, ext_ax), (co_ct, co_d, ext_co), (sa_ct, sa_d, ext_sa) = \
            extract_triplanar_arrays(ct, dose_r, pick=pick_method)
        
        overlay_panel(axs[row_idx, 0], ax_ct, ax_d, ext_ax, ct_wl, ct_ww, dose_min, dose_max, alpha, thresh)
        overlay_panel(axs[row_idx, 1], co_ct, co_d, ext_co, ct_wl, ct_ww, dose_min, dose_max, alpha, thresh)
        overlay_panel(axs[row_idx, 2], sa_ct, sa_d, ext_sa, ct_wl, ct_ww, dose_min, dose_max, alpha, thresh)
        
        if row_idx == 0:
            axs[row_idx, 0].set_title("Axial")
            axs[row_idx, 1].set_title("Coronal") 
            axs[row_idx, 2].set_title("Sagittal")

    fig.suptitle(title, fontsize=14)

    if output:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/{title}.png", dpi=200)
        print(f"Saved: {title}.png")
    else:
        plt.show()

    plt.close(fig)
