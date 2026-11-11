"""Recalculate cumulative DVHs from RTSTRUCT contours and a DICOM dose grid.

A CT-centred slab approximation is used, not a treatment-planning-system DVH.
See README for sampling, coverage, contour, and dose interpretation limits.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .triplanar import _read_dose, _read_ct, _check_references, _identity, _require


def list_rois(rtstruct_path):
    """List ROI numbers, exact names, and contour types without reading pixels."""
    import pydicom
    struct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    _require(struct.get('Modality') == 'RTSTRUCT', 'Expected an RTSTRUCT file.')
    contours = {int(c.ReferencedROINumber): c for c in struct.get('ROIContourSequence', [])}
    return [
        {'number': int(r.ROINumber), 'name': str(r.get('ROIName', '')),
         'contour_types': sorted({str(c.ContourGeometricType) for c in
                                 contours.get(int(r.ROINumber), {}).get('ContourSequence', [])}),
         'contour_count': len(contours.get(int(r.ROINumber), {}).get('ContourSequence', []))}
        for r in struct.get('StructureSetROISequence', [])
    ]


def _select_rois(struct, roi_numbers, roi_names):
    definitions = list(struct.get('StructureSetROISequence', []))
    lookup = {int(r.ROINumber): r for r in definitions}
    _require(len(lookup) == len(definitions), 'Duplicate ROI numbers in RTSTRUCT.')
    _require(roi_numbers or roi_names, 'Select roi_numbers or exact roi_names; use list_rois first.')
    numbers = []
    for number in roi_numbers or []:
        _require(isinstance(number, int) and not isinstance(number, bool) and number in lookup,
                 f'Unknown or invalid ROI number: {number!r}.')
        numbers.append(number)
    for name in roi_names or []:
        matches = [int(r.ROINumber) for r in definitions if str(r.get('ROIName', '')) == name]
        _require(len(matches) == 1, f'ROI name {name!r} is missing or ambiguous; select by number.')
        numbers.append(matches[0])
    contours = list(struct.get('ROIContourSequence', []))
    contour_lookup = {int(c.ReferencedROINumber): c for c in contours}
    _require(len(contour_lookup) == len(contours), 'Duplicate ROIContourSequence entries.')
    result = []
    for number in dict.fromkeys(numbers):
        _require(number in contour_lookup, f'ROI {number} has no contours.')
        result.append((lookup[number], contour_lookup[number]))
    return result


def _inside_polygon(points, polygon):
    """Even/odd polygon fill, including its boundary; also supports keyholes."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    boundary = inside.copy()
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = b - a
        length = np.hypot(dx, dy)
        if length <= 1e-12:
            continue
        cross = (x-a[0])*dy - (y-a[1])*dx
        boundary |= ((np.abs(cross) <= 1e-8*length)
                     & (x >= min(a[0], b[0])-1e-8) & (x <= max(a[0], b[0])+1e-8)
                     & (y >= min(a[1], b[1])-1e-8) & (y <= max(a[1], b[1])+1e-8))
        if dy != 0:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < a[0] + (y-a[1])*dx/dy)
    return inside | boundary


def _contour_planes(definition, roi, ct):
    number = int(definition.ROINumber)
    _require(str(definition.get('ReferencedFrameOfReferenceUID', '')) == ct.metadata['FrameOfReferenceUID'],
             f'ROI {number}: missing or mismatched ReferencedFrameOfReferenceUID.')
    _require(not roi.get('SourcePixelPlanesCharacteristicsSequence'),
             f'ROI {number}: SourcePixelPlanesCharacteristicsSequence is not supported yet.')
    contours = list(roi.get('ContourSequence', []))
    types = {str(c.get('ContourGeometricType', '')) for c in contours}
    _require(types in ({'CLOSED_PLANAR'}, {'CLOSEDPLANAR_XOR'}),
             f'ROI {number}: only unmixed CLOSED_PLANAR or CLOSEDPLANAR_XOR contours are supported.')
    planes, missing_refs = {}, 0
    for contour in contours:
        _require('ContourSlabThickness' not in contour and 'ContourOffsetVector' not in contour,
                 f'ROI {number}: explicit contour slabs/offsets are not supported.')
        values = np.asarray(contour.get('ContourData', []), dtype=float)
        count = int(contour.get('NumberOfContourPoints', 0))
        _require(count >= 3 and values.size == count*3 and np.isfinite(values).all(),
                 f'ROI {number}: invalid contour coordinates or point count.')
        xyz = values.reshape(-1, 3)
        local = (xyz-ct.origin) @ ct.basis
        z = float(local[:, 2].mean())
        index = int(np.argmin(np.abs(ct.offsets-z)))
        _require(np.max(np.abs(local[:, 2]-ct.offsets[index])) <= 0.01,
                 f'ROI {number}: contour is nonplanar or not on a selected CT plane (0.01 mm tolerance).')
        bounds = np.array([[-ct.col_spacing/2, ct.cols[-1]+ct.col_spacing/2],
                           [-ct.row_spacing/2, ct.rows[-1]+ct.row_spacing/2]])
        _require(np.all(local[:, :2] >= bounds[:, 0]-1e-6)
                 and np.all(local[:, :2] <= bounds[:, 1]+1e-6),
                 f'ROI {number}: contour extends outside the CT field of view; refusing to clip its volume.')
        refs = contour.get('ContourImageSequence', [])
        for ref in refs:
            _require(str(ref.get('ReferencedSOPInstanceUID', '')) == ct.metadata['slice_sop_instance_uids'][index],
                     f'ROI {number}: contour image reference does not match its CT plane.')
        missing_refs += not bool(refs)
        planes.setdefault(index, []).append(local[:, :2])
    return planes, next(iter(types)), missing_refs


def _sample_roi(definition, roi, ct, dose, subsamples):
    planes, kind, missing_refs = _contour_planes(definition, roi, ct)
    samples, count = [], 0
    # Each CT pixel is split into equally weighted in-plane subpixels.
    xaxis = ((np.arange(ct.data.shape[2]*subsamples)+0.5)/subsamples-0.5)*ct.col_spacing
    yaxis = ((np.arange(ct.data.shape[1]*subsamples)+0.5)/subsamples-0.5)*ct.row_spacing
    for index, polygons in sorted(planes.items()):
        vertices = np.concatenate(polygons)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        xs = xaxis[(xaxis >= lo[0]-1e-8) & (xaxis <= hi[0]+1e-8)]
        ys = yaxis[(yaxis >= lo[1]-1e-8) & (yaxis <= hi[1]+1e-8)]
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        mask = np.zeros(len(points), dtype=bool)
        for polygon in polygons:
            inside = _inside_polygon(points, polygon)
            if kind == 'CLOSEDPLANAR_XOR':
                mask ^= inside
            else:
                mask |= inside
        points = points[mask]
        count += len(points)
        local = np.column_stack([points, np.full(len(points), ct.offsets[index])])
        values = dose.sample(ct.origin + local @ ct.basis.T)
        samples.append(values[np.isfinite(values)])
    values = np.concatenate(samples)
    _require(count > 0, f'ROI {int(definition.ROINumber)} contains no sampled points; increase in_plane_subsamples.')
    # CT geometry validation requires uniformly spaced slices. Empty planes are
    # left empty: no contour interpolation or extension across gaps is performed.
    thickness = float(np.median(np.diff(ct.offsets)))
    weight_cc = ct.row_spacing*ct.col_spacing*thickness/(1000*subsamples**2)
    missing_planes = sorted(set(range(min(planes), max(planes)+1))-set(planes))
    return values, count, weight_cc, {
        'contour_type': kind, 'contoured_ct_planes': sorted(planes),
        'uncontoured_planes_between_extents': missing_planes,
        'contours_without_image_reference': missing_refs,
    }


def _metrics(values, weight_cc, v_doses):
    if len(values) == 0:
        return None
    ordered = np.sort(values)
    count = len(values)
    result = {'Dmin_Gy': float(ordered[0]), 'Dmax_Gy': float(ordered[-1]),
              'Dmean_Gy': float(np.mean(values, dtype=np.float64))}
    # Dp is the greatest sampled dose with >= p% of equally weighted samples
    # at or above it. It is calculated from samples, never histogram bins.
    for percent in (2, 50, 95, 98):
        result[f'D{percent}_Gy'] = float(ordered[count-math.ceil(count*percent/100)])
    result['V_dose'] = [
        {'dose_Gy': threshold, 'volume_cc': int(np.count_nonzero(values >= threshold))*weight_cc,
         'volume_pct': 100*int(np.count_nonzero(values >= threshold))/count}
        for threshold in v_doses
    ]
    return result


def _fingerprint(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def calculate_dvh(ct_path, rtdose_path, rtstruct_path, *, roi_numbers=None, roi_names=None,
                  rtplan_path=None, ct_series_uid=None, in_plane_subsamples=2,
                  bin_width_gy=0.1, v_doses_gy=None, allow_partial=False, workers=4):
    """Return JSON-ready cumulative DVHs and metrics for explicitly selected ROIs.

    Requires dose units GY, matching patient/study/frame, and contours on the
    selected conventional CT planes. DoseType and DoseSummationType are retained.
    Out-of-grid samples raise unless allow_partial=True; then whole-ROI metrics
    are withheld and the covered-only DVH is clearly labelled. No dose summation,
    normalization, registration, or prescription acceptance is performed.
    """
    import pydicom
    _require(isinstance(in_plane_subsamples, int) and not isinstance(in_plane_subsamples, bool)
             and 1 <= in_plane_subsamples <= 8, 'in_plane_subsamples must be an integer from 1 to 8.')
    _require(isinstance(workers, int) and not isinstance(workers, bool) and workers > 0,
             'workers must be a positive integer.')
    _require(np.isfinite(bin_width_gy) and bin_width_gy > 0, 'bin_width_gy must be positive and finite.')
    thresholds = sorted(set(float(v) for v in (v_doses_gy or [])))
    _require(all(np.isfinite(v) and v >= 0 for v in thresholds), 'V-dose thresholds must be finite and nonnegative.')
    dose, ds = _read_dose(rtdose_path, allow_zero=True)
    _require(ds.DoseUnits == 'GY', 'DVH Gy metrics require DoseUnits GY; RELATIVE dose is not converted.')
    _require(ds.get('DoseType') in ('PHYSICAL', 'EFFECTIVE'),
             'DVH requires PHYSICAL or EFFECTIVE DoseType; error/unspecified dose is unsupported.')
    ct, ct_sops = _read_ct(ct_path, dose, ct_series_uid, workers)
    checks, notes = _check_references(dose, ds, ct, ct_sops, rtplan_path, rtstruct_path)
    struct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    rois = []
    for definition, contour in _select_rois(struct, roi_numbers, roi_names):
        values, count, weight, geometry = _sample_roi(definition, contour, ct, dose, in_plane_subsamples)
        complete = len(values) == count
        number = int(definition.ROINumber)
        _require(complete or allow_partial,
                 f'ROI {number}: {count-len(values)}/{count} samples outside RTDOSE; use allow_partial=True for covered-only results.')
        ordered = np.sort(values)
        maximum = float(ordered[-1]) if len(values) else 0
        bins = math.ceil(maximum/bin_width_gy)+2
        _require(bins <= 100000, 'bin_width_gy creates more than 100000 dose thresholds.')
        doses = np.arange(bins)*bin_width_gy
        volumes = (len(values)-np.searchsorted(ordered, doses, side='left'))*weight
        metrics = _metrics(values, weight, thresholds)
        warnings = []
        if not complete:
            warnings.append('Partial dose coverage: curve and covered_metrics describe only covered samples; whole-ROI metrics are withheld.')
        if geometry['uncontoured_planes_between_extents']:
            warnings.append('Uncontoured CT planes between ROI extents are empty; no interpolation across gaps.')
        if geometry['contours_without_image_reference']:
            warnings.append('Some contours lack image references; CT plane assignment uses geometry only.')
        rois.append({
            'number': number, 'name': str(definition.get('ROIName', '')),
            'coverage': 'complete' if complete else 'partial',
            'sample_count': count, 'covered_sample_count': len(values),
            'sampled_volume_cc': count*weight, 'covered_volume_cc': len(values)*weight,
            'uncovered_volume_cc': (count-len(values))*weight,
            'coverage_pct': 100*len(values)/count,
            'metrics': metrics if complete else None, 'covered_metrics': metrics,
            'geometry': geometry, 'warnings': warnings,
            'cumulative': {'dose_Gy': doses.tolist(), 'volume_cc': volumes.tolist(),
                           'volume_pct': (100*volumes/(len(values)*weight)).tolist() if len(values) else None,
                           'percent_denominator': 'covered_volume_cc'},
        })
    paths = {'rtdose': rtdose_path, 'rtstruct': rtstruct_path}
    if rtplan_path is not None:
        paths['rtplan'] = rtplan_path
    return {
        'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'calculation': 'dose_view CT-slab DVH v1', 'clinical_acceptance': 'not_assessed',
        'sources': {name: _fingerprint(path) for name, path in paths.items()},
        'ct': ct.summary(), 'dose': dose.summary(), 'rtstruct': _identity(struct),
        'checks': checks, 'reference_metadata_notes': notes,
        'sampling': {
            'in_plane_subsamples': in_plane_subsamples, 'bin_width_Gy': bin_width_gy,
            'plane_tolerance_mm': 0.01, 'ct_slab_thickness_mm': float(np.median(np.diff(ct.offsets))),
            'volume_model': 'Each contoured CT plane represents one full CT slice-spacing slab; empty planes remain empty.',
            'dose_sampling': 'Trilinear dose at in-plane subpixel centres and CT plane centres; no through-plane supersampling.',
            'contour_fill': 'Even-odd within each polygon (boundary included); union CLOSED_PLANAR, XOR CLOSEDPLANAR_XOR.',
            'D_percent_definition': 'Greatest sampled dose for which at least the requested percent of samples receive that dose or more.',
            'coverage_definition': 'Coverage of sampled centres within the RTDOSE interpolation domain, not all points in each slab.',
        },
        'rois': rois,
    }


def save_dvh(ct_path, rtdose_path, rtstruct_path, output_path, *, title=None, dpi=160, **kwargs):
    """Calculate and save a PNG, .dvh.json report, and .dvh.csv curves.

    Existing files are never overwritten. For calculation options see
    calculate_dvh. All outputs may contain patient identifiers and source paths.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _require(isinstance(dpi, int) and not isinstance(dpi, bool) and dpi > 0, 'dpi must be a positive integer.')
    output = Path(output_path).expanduser().resolve()
    _require(output.suffix.lower() == '.png', 'output_path must end in .png.')
    paths = (output, output.with_suffix('.dvh.json'), output.with_suffix('.dvh.csv'))
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    report = calculate_dvh(ct_path, rtdose_path, rtstruct_path, **kwargs)
    report['outputs'] = dict(zip(('figure', 'report', 'curves'), map(str, paths)))
    fig = Figure(figsize=(10, 7), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for roi in report['rois']:
        curve = roi['cumulative']
        partial = roi['coverage'] != 'complete'
        label = f"{roi['number']}: {roi['name']} ({roi['sampled_volume_cc']:.3g} cc)"
        if partial:
            label += f" — PARTIAL {roi['coverage_pct']:.1f}% covered"
        if curve['volume_pct'] is not None:
            ax.step(curve['dose_Gy'], curve['volume_pct'], where='pre',
                    ls='--' if partial else '-', label=label)
    ax.set(xlabel=f"Dose (Gy; DICOM {report['dose']['DoseType']})",
           ylabel='Volume receiving ≥ dose (% of covered volume)', ylim=(0, 102), xlim=(0, None))
    ax.set_title(title or 'Cumulative dose–volume histogram', loc='left', weight='bold')
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend(fontsize=8, loc='best')
    partial = [r for r in report['rois'] if r['coverage'] != 'complete']
    footer = f"CT-slab approximation • DICOM dose scope: {report['dose']['DoseSummationType']} • Clinical acceptance not assessed"
    if partial:
        footer += '\nPARTIAL coverage: ' + ', '.join(f"ROI {r['number']} {r['coverage_pct']:.1f}%" for r in partial)
    fig.supxlabel(footer, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open('xb') as stream:
            created.append(output)
            fig.savefig(stream, format='png', dpi=dpi)
        with paths[1].open('x') as stream:
            created.append(paths[1])
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        with paths[2].open('x', newline='') as stream:
            created.append(paths[2])
            writer = csv.writer(stream)
            writer.writerow(['roi_number', 'roi_name', 'coverage', 'coverage_pct',
                             'dose_Gy', 'volume_cc', 'volume_pct_of_covered'])
            for roi in report['rois']:
                c = roi['cumulative']
                for dose, volume, percent in zip(c['dose_Gy'], c['volume_cc'],
                                                c['volume_pct'] or ['']*len(c['dose_Gy'])):
                    writer.writerow([roi['number'], roi['name'], roi['coverage'],
                                     roi['coverage_pct'], dose, volume, percent])
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        fig.clear()
    return report
