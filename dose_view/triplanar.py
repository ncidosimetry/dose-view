"""DICOM CT/dose overlays in the patient LPS coordinate system.

Optional dependencies are installed with ``pip install 'dose_view[dicom]'``.
The existing notebook plotting functions remain available in ``visualization``.
"""
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _basis(orientation):
    orientation = np.asarray(orientation, dtype=float)
    _require(orientation.shape == (6,), 'ImageOrientationPatient must contain six values.')
    col, row = orientation[:3], orientation[3:]
    basis = np.column_stack((col, row, np.cross(col, row)))
    _require(np.isfinite(basis).all() and np.allclose(basis.T @ basis, np.eye(3), atol=1e-5),
             'ImageOrientationPatient must describe orthonormal image axes.')
    return basis


def _geometry(ds):
    origin = np.asarray(ds.get('ImagePositionPatient', []), dtype=float)
    _require(origin.shape == (3,) and np.isfinite(origin).all(),
             'ImagePositionPatient must contain three finite coordinates.')
    spacing = np.asarray(ds.get('PixelSpacing', []), dtype=float)
    _require(spacing.shape == (2,) and np.isfinite(spacing).all() and np.all(spacing > 0),
             'PixelSpacing must contain two positive finite values.')
    return origin, _basis(ds.get('ImageOrientationPatient', [])), spacing


@dataclass
class _Volume:
    """Pixel array indexed (frame, row, column), with geometry in mm."""
    data: np.ndarray
    origin: np.ndarray
    basis: np.ndarray
    row_spacing: float
    col_spacing: float
    offsets: np.ndarray
    metadata: dict

    def __post_init__(self):
        from scipy.interpolate import RegularGridInterpolator
        _require(self.data.ndim == 3 and min(self.data.shape) >= 2,
                 'Triplanar rendering requires at least two voxels along each image axis.')
        _require(self.origin.shape == (3,) and np.isfinite(self.origin).all(),
                 'ImagePositionPatient must contain three finite coordinates.')
        _require(np.isfinite([self.row_spacing, self.col_spacing]).all()
                 and self.row_spacing > 0 and self.col_spacing > 0, 'PixelSpacing must be positive and finite.')
        _require(self.offsets.shape == (self.data.shape[0],) and np.isfinite(self.offsets).all(),
                 'Frame positions must be finite and match the pixel array.')
        _require(np.all(np.diff(self.offsets) > 0), 'Frame positions must be strictly increasing.')
        self.rows = np.arange(self.data.shape[1]) * self.row_spacing
        self.cols = np.arange(self.data.shape[2]) * self.col_spacing
        self._interpolator = RegularGridInterpolator(
            (self.offsets, self.rows, self.cols), self.data,
            method='linear', bounds_error=False, fill_value=np.nan)

    def sample(self, patient_xyz):
        local = (np.asarray(patient_xyz) - self.origin) @ self.basis
        # Avoid masking exact edge coordinates because of round-off after rotation.
        for i, axis in enumerate((self.cols, self.rows, self.offsets)):
            for edge in (axis[0], axis[-1]):
                local[..., i] = np.where(np.isclose(local[..., i], edge, rtol=0, atol=1e-7), edge, local[..., i])
        return self._interpolator(local[..., [2, 1, 0]])

    def bounds(self):
        corners = np.array(list(product(
            (self.cols[0], self.cols[-1]), (self.rows[0], self.rows[-1]),
            (self.offsets[0], self.offsets[-1]))))
        xyz = self.origin + corners @ self.basis.T
        return np.stack((xyz.min(axis=0), xyz.max(axis=0)), axis=1)

    def summary(self):
        return dict(self.metadata, shape_zyx=list(self.data.shape), origin_xyz_mm=self.origin.tolist(),
                    direction_xyz=self.basis.tolist(), pixel_spacing_yx_mm=[self.row_spacing, self.col_spacing],
                    frame_offsets_mm=self.offsets.tolist(), bounds_xyz_mm=self.bounds().tolist())


def _identity(ds):
    return {key: str(ds.get(key, '')) for key in
            ('PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'FrameOfReferenceUID', 'SOPInstanceUID', 'SOPClassUID')}


def _check_identity(reference, other, label):
    for key in ('PatientID', 'StudyInstanceUID', 'FrameOfReferenceUID'):
        _require(bool(reference.get(key)) and reference[key] == other.get(key),
                 f'{label}: missing or different {key}; an unregistered overlay is not allowed.')


def _read_dose(path):
    import pydicom
    ds = pydicom.dcmread(path)
    _require(ds.get('Modality') == 'RTDOSE', 'rtdose_path must identify an RTDOSE DICOM file.')
    _require('DoseGridScaling' in ds, 'RTDOSE is missing DoseGridScaling.')
    scaling = float(ds.DoseGridScaling)
    _require(np.isfinite(scaling) and scaling > 0, 'DoseGridScaling must be positive and finite.')
    _require(ds.get('DoseUnits') in ('GY', 'RELATIVE'), 'Supported DoseUnits are GY and RELATIVE.')
    data = ds.pixel_array.astype(np.float32) * scaling
    _require(data.ndim == 3, 'RTDOSE must be a multi-frame dose volume.')
    _require(np.isfinite(data).all() and data.min() >= 0 and data.max() > 0,
             'RTDOSE must contain finite, nonnegative values and some positive dose.')
    origin, basis, spacing = _geometry(ds)
    offsets = np.asarray(ds.get('GridFrameOffsetVector', []), dtype=float)
    _require(offsets.shape == (data.shape[0],) and np.isfinite(offsets).all(),
             'GridFrameOffsetVector must contain one finite offset per frame.')
    if np.isclose(offsets[0], 0, rtol=0, atol=1e-6):
        interpretation = 'relative offsets along the image normal'
    elif np.allclose(basis, np.eye(3), rtol=0, atol=1e-6) and np.isclose(offsets[0], origin[2], rtol=0, atol=1e-6):
        offsets = offsets - origin[2]
        interpretation = 'absolute patient z coordinates, converted to relative offsets'
    else:
        raise ValueError('GridFrameOffsetVector does not follow a supported DICOM relative/absolute convention.')
    if np.all(np.diff(offsets) < 0):
        offsets, data = offsets[::-1], data[::-1]
    metadata = dict(_identity(ds), path=str(Path(path).resolve()), DoseGridScaling=scaling,
                    DoseUnits=str(ds.DoseUnits), DoseType=str(ds.get('DoseType', 'UNSPECIFIED')),
                    DoseSummationType=str(ds.get('DoseSummationType', 'UNSPECIFIED')),
                    grid_frame_offset_interpretation=interpretation)
    return _Volume(data, origin, basis, *spacing, offsets, metadata), ds


def _read_ct(path, dose, series_uid, workers):
    import pydicom
    path = Path(path)
    _require(path.is_dir(), 'ct_path must be a directory containing a DICOM CT series.')
    files = sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith('.'))

    def header(file):
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            return None
        return (file, ds) if ds.get('Modality') == 'CT' else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        headers = [h for h in pool.map(header, files) if h is not None]
    series = {str(ds.get('SeriesInstanceUID', '')) for _, ds in headers}
    if series_uid is None:
        _require(len(series) == 1, f'CT directory contains {len(series)} series; supply ct_series_uid to select one.')
        series_uid = next(iter(series))
    headers = [(file, ds) for file, ds in headers if str(ds.get('SeriesInstanceUID', '')) == series_uid]
    _require(len(headers) >= 2, 'The selected CT series must contain at least two slices.')
    first = headers[0][1]
    _, basis, spacing = _geometry(first)
    positions = []
    for _, ds in headers:
        _check_identity(dose.metadata, _identity(ds), 'CT/RTDOSE')
        _require(bool(ds.get('SOPInstanceUID')) and bool(ds.get('SeriesInstanceUID')),
                 'CT slices must have SOPInstanceUID and SeriesInstanceUID.')
        _require(int(ds.get('NumberOfFrames', 1)) == 1 and 'PerFrameFunctionalGroupsSequence' not in ds,
                 'Only conventional single-frame CT slices are supported.')
        position, slice_basis, slice_spacing = _geometry(ds)
        _require(np.allclose(slice_basis, basis, rtol=0, atol=1e-5), 'CT slice orientations differ.')
        _require(np.allclose(slice_spacing, spacing, rtol=0, atol=1e-6), 'CT pixel spacing differs across slices.')
        _require((int(ds.Rows), int(ds.Columns)) == (int(first.Rows), int(first.Columns)), 'CT slice sizes differ.')
        _require(ds.get('PhotometricInterpretation') == 'MONOCHROME2', 'CT must use MONOCHROME2 pixels.')
        _require('RescaleSlope' in ds and 'RescaleIntercept' in ds,
                 'CT slices must provide RescaleSlope and RescaleIntercept for HU conversion.')
        _require(str(ds.get('RescaleType', 'HU')) == 'HU', 'CT rescale units must be HU.')
        positions.append(position)
    positions = np.asarray(positions)
    order = np.argsort(positions @ basis[:, 2])
    headers = [headers[i] for i in order]
    positions = positions[order]
    origin = positions[0]
    local_positions = (positions-origin) @ basis
    _require(np.allclose(local_positions[:, :2], 0, atol=1e-3), 'CT slices have in-plane shifts; resample that CT series first.')
    offsets = local_positions[:, 2]
    _require(np.all(np.diff(offsets) > 1e-5), 'CT contains duplicate or unordered slice positions.')
    _require(np.allclose(np.diff(offsets), np.median(np.diff(offsets)), rtol=1e-3, atol=1e-3),
             'CT has missing slices or nonuniform spacing; resolve the series before plotting.')
    sop_uids = {str(ds.SOPInstanceUID) for _, ds in headers}
    _require(len(sop_uids) == len(headers), 'CT contains duplicate SOP Instance UIDs.')

    def pixels(item):
        ds = pydicom.dcmread(item[0])
        data = ds.pixel_array.astype(np.float32)
        return data * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        data = np.stack(list(pool.map(pixels, headers)))
    _require(np.isfinite(data).all(), 'Rescaled CT values must be finite.')
    metadata = dict(_identity(first), path=str(path.resolve()), slice_count=len(headers))
    metadata.pop('SOPInstanceUID')  # This volume has a series UID, not a single SOP UID.
    return _Volume(data, origin, basis, *spacing, offsets, metadata), sop_uids


def _check_references(dose, ds, ct, ct_sops, rtplan_path, rtstruct_path):
    import pydicom
    checks = {'same_patient_study_and_frame': True, 'referenced_ct_images_checked': False}
    notes = []
    plan = struct = None
    if rtplan_path is not None:
        plan = pydicom.dcmread(rtplan_path, stop_before_pixels=True)
        _require(plan.get('Modality') == 'RTPLAN', 'rtplan_path must identify an RTPLAN file.')
        _check_identity(dose.metadata, _identity(plan), 'RTPLAN/RTDOSE')
        refs = [r for r in ds.get('ReferencedRTPlanSequence', []) if str(r.ReferencedSOPInstanceUID) == str(plan.SOPInstanceUID)]
        _require(bool(refs), 'RTDOSE does not reference the supplied RTPLAN SOP Instance UID.')
        checks['dose_to_plan_instance_uid_match'] = True
        if any(str(r.get('ReferencedSOPClassUID', '')) != str(plan.SOPClassUID) for r in refs):
            notes.append('Dose-to-plan SOP Instance UID matches, but its ReferencedSOPClassUID differs from the actual plan class.')
    if rtstruct_path is not None:
        struct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        _require(struct.get('Modality') == 'RTSTRUCT', 'rtstruct_path must identify an RTSTRUCT file.')
        identity = _identity(struct)
        frames = struct.get('ReferencedFrameOfReferenceSequence', [])
        # RTSTRUCT may express its frame only in the referenced-frame sequence.
        frame_uids = {str(f.FrameOfReferenceUID) for f in frames}
        _require(dose.metadata['FrameOfReferenceUID'] in frame_uids, 'RTSTRUCT does not reference the CT/dose frame.')
        identity['FrameOfReferenceUID'] = dose.metadata['FrameOfReferenceUID']
        _check_identity(dose.metadata, identity, 'RTSTRUCT/RTDOSE')
        if plan is not None:
            _require(str(struct.SOPInstanceUID) in {str(r.ReferencedSOPInstanceUID) for r in plan.get('ReferencedStructureSetSequence', [])},
                     'RTPLAN does not reference the supplied RTSTRUCT SOP Instance UID.')
            checks['plan_to_structure_instance_uid_match'] = True
        selected_refs = []
        for frame in frames:
            if str(frame.FrameOfReferenceUID) != dose.metadata['FrameOfReferenceUID']:
                continue
            for study in frame.get('RTReferencedStudySequence', []):
                for series in study.get('RTReferencedSeriesSequence', []):
                    if str(series.SeriesInstanceUID) == ct.metadata['SeriesInstanceUID']:
                        selected_refs.append(series)
        _require(bool(selected_refs), 'RTSTRUCT does not reference the selected CT series.')
        image_refs = {str(r.ReferencedSOPInstanceUID) for s in selected_refs for r in s.get('ContourImageSequence', [])}
        _require(image_refs.issubset(ct_sops), 'Some RTSTRUCT-referenced CT slices are missing from the selected series.')
        checks.update(structure_to_ct_series_match=True, referenced_ct_images_present=len(image_refs),
                      referenced_ct_images_checked=bool(image_refs))
        if not image_refs:
            notes.append('RTSTRUCT references the CT series but lists no CT image instance UIDs; slice availability was not checked.')
    checks['plan_reference_checked'] = plan is not None
    checks['structure_reference_checked'] = struct is not None
    checks['plan_to_structure_reference_checked'] = plan is not None and struct is not None
    return checks, notes


def _center(dose, ct, pick, center_xyz_mm, fraction):
    if center_xyz_mm is not None:
        center = np.asarray(center_xyz_mm, dtype=float)
        method = 'explicit patient coordinates'
    elif pick == 'dose_center':
        mask = dose.data >= fraction * dose.data.max()
        indices = np.nonzero(mask)
        weights = dose.data[mask]
        coords = [np.average(axis[ind], weights=weights) for axis, ind in
                  zip((dose.offsets, dose.rows, dose.cols), indices)]
        center = dose.origin + np.array(coords[::-1]) @ dose.basis.T
        method = f'dose-weighted center of voxels >= {fraction * 100:g}% of maximum'
    elif pick == 'maxdose':
        iz, iy, ix = np.unravel_index(dose.data.argmax(), dose.data.shape)
        center = dose.origin + np.array([dose.cols[ix], dose.rows[iy], dose.offsets[iz]]) @ dose.basis.T
        method = 'maximum native dose voxel'
    elif pick == 'center':
        local = np.array([ct.cols[-1]/2, ct.rows[-1]/2, (ct.offsets[0]+ct.offsets[-1])/2])
        center = ct.origin + local @ ct.basis.T
        method = 'CT volume center'
    else:
        raise ValueError('pick must be dose_center, maxdose, or center.')
    _require(center.shape == (3,) and np.isfinite(center).all(), 'center_xyz_mm must contain three finite coordinates.')
    _require(np.isfinite(ct.sample(center)).all(), 'Selected center is outside the CT; supply center_xyz_mm within the CT.')
    return center, method


def _planes(ct, dose, center, crop_mm):
    bounds = ct.bounds()
    if crop_mm is not None:
        _require(isinstance(crop_mm, Mapping), 'crop_mm must map x/y/z to coordinate limits.')
        _require(set(crop_mm).issubset({'x', 'y', 'z'}), 'crop_mm keys must be x, y, or z.')
        for i, key in enumerate('xyz'):
            if key in crop_mm:
                interval = np.asarray(crop_mm[key], dtype=float)
                _require(interval.shape == (2,) and np.isfinite(interval).all() and interval[0] < interval[1], 'Crop limits must be finite [minimum, maximum] pairs.')
                bounds[i] = [max(bounds[i, 0], interval[0]), min(bounds[i, 1], interval[1])]
    _require(np.all(bounds[:, 0] < bounds[:, 1]), 'Crop does not overlap the CT volume.')
    _require(np.all(center >= bounds[:, 0]) and np.all(center <= bounds[:, 1]), 'Crop excludes the selected crosshair.')
    native_spacing = np.array([ct.col_spacing, ct.row_spacing, np.median(np.diff(ct.offsets))])
    # Sampling distance along each patient axis, including for an oblique CT.
    step = 1 / np.linalg.norm(ct.basis / native_spacing, axis=1)
    axes = [np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)/s))+1)) for (lo,hi),s in zip(bounds,step)]
    panels = []
    for name, horizontal, vertical, fixed, labels in (
        ('Axial', 0, 1, 2, ('R','L','A','P')),
        ('Sagittal', 1, 2, 0, ('A','P','S','I')),
        ('Coronal', 0, 2, 1, ('R','L','S','I')),
    ):
        h, v = axes[horizontal], axes[vertical]
        hh, vv = np.meshgrid(h, v)
        points = np.empty(hh.shape + (3,))
        points[..., horizontal], points[..., vertical], points[..., fixed] = hh, vv, center[fixed]
        ct2d, dose2d = ct.sample(points), dose.sample(points)
        # Do not paint dose beyond the sampled CT domain on an oblique reformat.
        dose2d[~np.isfinite(ct2d)] = np.nan
        dh, dv = h[1]-h[0], v[1]-v[0]
        panels.append(dict(name=name, ct=ct2d, dose=dose2d, extent=[h[0]-dh/2,h[-1]+dh/2,v[0]-dv/2,v[-1]+dv/2],
                           horizontal=horizontal, vertical=vertical, fixed=fixed, orientation=labels))
    _require(any(np.any(p['dose'] > 0) for p in panels), 'The selected planes contain no positive dose within the CT.')
    return panels


def _render(panels, center, metadata, output_path, title, subtitle, ct_wl, ct_ww, alpha, cmap, dpi):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    maximum, threshold = metadata['dose_max'], metadata['display']['dose_threshold']
    units = 'Gy' if metadata['dose']['DoseUnits'] == 'GY' else 'relative units'
    fig = Figure(figsize=(17, 10), facecolor='#101721')
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(1,4,width_ratios=[1.15,.90,1.15,.055],left=.055,right=.947,bottom=.14,top=.82,wspace=.22)
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0,index], facecolor='black')
        extent = panel['extent']
        ax.imshow(panel['ct'],cmap='gray',vmin=ct_wl-ct_ww/2,vmax=ct_wl+ct_ww/2,extent=extent,origin='lower',interpolation='nearest')
        masked = np.ma.masked_where(~np.isfinite(panel['dose']) | (panel['dose'] < threshold),panel['dose'])
        ax.imshow(masked,cmap=cmap,vmin=0,vmax=maximum,alpha=alpha,extent=extent,origin='lower',interpolation='nearest')
        if panel['name']=='Axial':
            ax.invert_yaxis()
        ax.axvline(center[panel['horizontal']],color='white',lw=.65,alpha=.55,ls=(0,(4,5)))
        ax.axhline(center[panel['vertical']],color='white',lw=.65,alpha=.55,ls=(0,(4,5)))
        fixed = 'xyz'[panel['fixed']]
        ax.set_title(f"{panel['name']}  |  {fixed} = {center[panel['fixed']]:.1f} mm",loc='left',pad=15,fontsize=13,weight='bold',color='#ecf1f7')
        ax.set_xlabel(f"Patient {'xyz'[panel['horizontal']]} (mm)",color='#c4d0dc',labelpad=8)
        ax.set_ylabel(f"Patient {'xyz'[panel['vertical']]} (mm)",color='#c4d0dc',labelpad=6)
        ax.tick_params(colors='#9facba')
        for spine in ax.spines.values():
            spine.set_color('#445368')
        ax.set_aspect('equal')
        ax.set_anchor('N')
        for label,pos in zip(panel['orientation'],[(.035,.5),(.965,.5),(.5,.965),(.5,.035)]):
            ax.text(*pos,label,transform=ax.transAxes,ha='center',va='center',color='white',weight='bold',fontsize=12,
                    bbox={'facecolor':'black','alpha':.55,'edgecolor':'none','pad':2})
    cax=fig.add_subplot(grid[0,3])
    bar=fig.colorbar(ScalarMappable(norm=Normalize(0,maximum),cmap=cmap),cax=cax)
    bar.set_label(f"Dose ({units}; DICOM {metadata['dose']['DoseType']})",color='#c4d0dc',labelpad=13)
    bar.ax.tick_params(colors='#9facba')
    bar.outline.set_edgecolor('#445368')
    bar.set_ticks(np.linspace(0,maximum,6))
    fig.text(.055,.947,title,fontsize=25,weight='bold',color='#ecf1f7')
    fig.text(.055,.905,subtitle,fontsize=13,color='#b9c9d9')
    fig.text(.055,.070,f'Dose maximum: {maximum:.2f} {units}   |   Overlay shown at >= {threshold:.2f} {units}   |   CT window: level {ct_wl:g}, width {ct_ww:g} HU',fontsize=11,color='#bbc9d7')
    fig.text(.055,.044,f"Linked planes: {metadata['display']['selection']}. Spatial scale is preserved within each panel.",fontsize=10,color='#97a8ba')
    fig.savefig(output_path,dpi=dpi,facecolor=fig.get_facecolor())
    fig.clear()


def save_triplanar_overlay(ct_path, rtdose_path, output_path, *, title=None, subtitle=None,
                           rtplan_path=None, rtstruct_path=None, ct_series_uid=None,
                           pick='dose_center', center_xyz_mm=None, crop_mm=None,
                           ct_wl=40, ct_ww=400, dose_threshold_fraction=0.10,
                           center_threshold_fraction=0.50, alpha=0.58, cmap='turbo', dpi=180, workers=8):
    """Save an axial/sagittal/coronal PNG and a ``.geometry.json`` sidecar.

    Inputs are a DICOM CT series directory and one RTDOSE file. Dose is scaled
    once using DoseGridScaling; no normalization or registration is applied.
    CT and dose must have the same patient, study, and frame of reference.
    Optional plan and structure files enable reference-chain checks.

    Planes intersect at ``center_xyz_mm`` in DICOM LPS millimeters, or at a
    point selected by ``pick``: ``dose_center``, ``maxdose``, or ``center``.
    ``crop_mm`` optionally maps x/y/z to increasing [minimum, maximum] limits.
    Threshold fractions are relative to the maximum of the entire dose grid.
    Unequal/oblique CT and dose directions and nonuniform dose frame offsets
    are supported; mixed or irregular CT series raise ValueError.

    Returns a JSON-serializable report including output paths, source UIDs,
    geometry, display settings, performed checks, and metadata inconsistencies.
    Existing output files at the explicitly supplied path are overwritten.
    """
    try:
        import pydicom  # noqa: F401
        import scipy.interpolate  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name not in ('pydicom', 'scipy', 'scipy.interpolate'):
            raise
        raise ImportError("Install DICOM plotting dependencies with pip install 'dose_view[dicom]'.") from error
    from matplotlib import colormaps
    _require(np.isfinite([ct_wl,ct_ww,alpha,dose_threshold_fraction,center_threshold_fraction]).all(), 'Display settings must be finite.')
    _require(ct_ww>0 and 0<alpha<=1, 'ct_ww must be positive and alpha must be in (0,1].')
    _require(0<=dose_threshold_fraction<1 and 0<center_threshold_fraction<=1, 'Threshold fractions must be in [0,1) for display and (0,1] for center selection.')
    _require(isinstance(workers,int) and not isinstance(workers,bool) and workers>0
             and isinstance(dpi,int) and not isinstance(dpi,bool) and dpi>0,
             'workers and dpi must be positive integers.')
    _require(isinstance(cmap, str) and cmap in colormaps, 'cmap must name a registered Matplotlib colormap.')
    output_path=Path(output_path).expanduser().resolve()
    _require(output_path.suffix.lower()=='.png', 'output_path must end in .png.')
    report_path=output_path.with_suffix('.geometry.json')
    for target in (output_path, report_path):
        if target.is_file():
            try:
                pydicom.dcmread(target, stop_before_pixels=True)
            except pydicom.errors.InvalidDicomError:
                pass
            else:
                raise ValueError('Output paths must not overwrite DICOM files.')
    dose,ds=_read_dose(rtdose_path)
    ct,ct_sops=_read_ct(ct_path,dose,ct_series_uid,workers)
    checks,notes=_check_references(dose,ds,ct,ct_sops,rtplan_path,rtstruct_path)
    center,selection=_center(dose,ct,pick,center_xyz_mm,center_threshold_fraction)
    panels=_planes(ct,dose,center,crop_mm)
    maximum=float(dose.data.max())
    metadata={
        'generated_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'figure':str(output_path),'geometry_report':str(report_path),
        'ct':ct.summary(),'dose':dose.summary(),'dose_max':maximum,
        'references':{'rtplan_path':str(Path(rtplan_path).resolve()) if rtplan_path is not None else None,
                      'rtstruct_path':str(Path(rtstruct_path).resolve()) if rtstruct_path is not None else None},
        'checks':dict(checks,ct_slice_geometry_checked=True,dose_frame_offsets_checked=True),
        'reference_metadata_notes':notes,
        'display':{'center_xyz_mm':center.tolist(),'selection':selection,
                   'crop_mm':{key:np.asarray(limits,dtype=float).tolist() for key,limits in crop_mm.items()} if crop_mm is not None else None,
                   'ct_window_level_HU':float(ct_wl),'ct_window_width_HU':float(ct_ww),
                   'dose_threshold':maximum*float(dose_threshold_fraction),'dose_threshold_fraction':float(dose_threshold_fraction),
                   'alpha':float(alpha),'cmap':cmap,'dpi':dpi,'interpolation':'trilinear in DICOM patient coordinates',
                   'outside_ct_or_dose_grid':'masked','extra_registration_or_renormalization':False},
        'geometry_reference':'https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.3.2.html',
    }
    title=title or f"{dose.metadata['PatientID']}  |  RT dose over CT"
    subtitle=subtitle or f"Selected {dose.metadata['DoseSummationType']} RTDOSE  /  {ct.data.shape[0]} CT slices"
    metadata['display'].update(title=title, subtitle=subtitle)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    _render(panels,center,metadata,output_path,title,subtitle,ct_wl,ct_ww,alpha,cmap,dpi)
    report_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata
