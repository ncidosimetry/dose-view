"""Small analytical DICOM fixtures; no patient data or network mounts required."""
from pathlib import Path
from copy import deepcopy
import builtins
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pydicom
from matplotlib.figure import Figure
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (CTImageStorage, RTDoseStorage, RTIonPlanStorage,
                         RTStructureSetStorage, ExplicitVRLittleEndian, generate_uid)

from dose_view import save_triplanar_overlay
from dose_view.triplanar import _Volume, _basis, _read_dose, _read_ct, _planes, _center


class TriplanarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ct_path = self.root / 'ct'
        self.ct_path.mkdir()
        self.study, self.frame, self.ct_series = generate_uid(), generate_uid(), generate_uid()
        for i, z in enumerate([10, 12, 14]):
            ds = self.dataset(CTImageStorage, 'CT')
            ds.SeriesInstanceUID = self.ct_series
            ds.ImagePositionPatient = [4, -2, z]
            ds.ImageOrientationPatient = [-1, 0, 0, 0, -1, 0]
            ds.PixelSpacing = [1, 1]
            ds.Rows, ds.Columns = 4, 5
            ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = 2, -100, 'HU'
            # CT world coordinates decrease with column/row, but HU is analytical.
            x, y = np.meshgrid(4-np.arange(5), -2-np.arange(4))
            stored = (100 + 2*x + 4*y + 6*z).astype('<i2')
            ds.PixelData = stored.tobytes()
            self.write_dicom(ds, self.ct_path / f'{2-i}.dcm')
        ds = self.dataset(RTDoseStorage, 'RTDOSE')
        ds.ImagePositionPatient = [0, -5, 10]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1, 1]
        ds.Rows, ds.Columns, ds.NumberOfFrames = 4, 5, 3
        ds.GridFrameOffsetVector = [0, 1, 4]
        ds.FrameIncrementPointer = [0x3004000C]
        ds.DoseGridScaling, ds.DoseUnits = 0.1, 'GY'
        ds.DoseType, ds.DoseSummationType = 'EFFECTIVE', 'PLAN'
        z, y, x = np.meshgrid([0, 1, 4], np.arange(4), np.arange(5), indexing='ij')
        ds.PixelData = ((20 + x + 2*y + 3*z)*10).astype('<i2').tobytes()
        self.dose_path = self.root / 'dose.dcm'
        self.write_dicom(ds)

    def tearDown(self):
        self.temp.cleanup()

    def write_dicom(self, ds, path=None):
        path = self.dose_path if path is None else path
        if int(pydicom.__version__.split('.')[0]) >= 3:
            ds.save_as(path, enforce_file_format=True)
        else:
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(path, write_like_original=False)

    def render(self, output_path=None, **kwargs):
        options = dict(workers=1, dpi=40)
        options.update(kwargs)
        return save_triplanar_overlay(
            self.ct_path, self.dose_path, output_path or self.root/'overlay.png', **options)

    def dataset(self, storage, modality):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = storage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
        ds.SOPClassUID, ds.SOPInstanceUID = storage, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.PatientID = modality, 'SYNTHETIC'
        ds.StudyInstanceUID, ds.FrameOfReferenceUID = self.study, self.frame
        ds.SeriesInstanceUID = generate_uid()
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        return ds

    def reference(self, ds):
        ref = Dataset()
        ref.ReferencedSOPInstanceUID, ref.ReferencedSOPClassUID = ds.SOPInstanceUID, ds.SOPClassUID
        return ref

    def reference_files(self):
        plan = self.dataset(RTIonPlanStorage, 'RTPLAN')
        struct = self.dataset(RTStructureSetStorage, 'RTSTRUCT')
        del struct.FrameOfReferenceUID
        series, study, frame = Dataset(), Dataset(), Dataset()
        series.SeriesInstanceUID = self.ct_series
        series.ContourImageSequence = [
            self.reference(pydicom.dcmread(path)) for path in sorted(self.ct_path.glob('*.dcm'))]
        study.RTReferencedSeriesSequence = [series]
        frame.FrameOfReferenceUID, frame.RTReferencedStudySequence = self.frame, [study]
        struct.ReferencedFrameOfReferenceSequence = [frame]
        plan.ReferencedStructureSetSequence = [self.reference(struct)]
        dose = pydicom.dcmread(self.dose_path)
        dose.ReferencedRTPlanSequence = [self.reference(plan)]
        self.plan_path, self.struct_path = self.root/'plan.dcm', self.root/'struct.dcm'
        self.write_dicom(plan, self.plan_path)
        self.write_dicom(struct, self.struct_path)
        self.write_dicom(dose)
        return plan, struct

    def test_flipped_ct_and_nonuniform_dose_use_world_coordinates(self):
        dose, _ = _read_dose(self.dose_path)
        ct, _ = _read_ct(self.ct_path, dose, None, 1)
        point = [2.5, -3.5, 12.5]
        self.assertAlmostEqual(float(dose.sample(point).item()), 20+2.5+2*1.5+3*2.5, places=5)
        self.assertAlmostEqual(float(ct.sample(point).item()), 2*(100+2*2.5+4*(-3.5)+6*12.5)-100, places=5)
        panels = _planes(ct, dose, np.array([2.,-4.,12.]), None)
        axial = panels[0]
        # Leftmost displayed column is the patient's right (low LPS x).
        self.assertAlmostEqual(float(axial['dose'][0,0]), 26)
        self.assertAlmostEqual(float(axial['dose'][-1,-1]), 36)

    def test_rotated_volume_interpolates_an_analytical_field(self):
        angle = np.pi/6
        basis = _basis([np.cos(angle),np.sin(angle),0,-np.sin(angle),np.cos(angle),0])
        origin = np.array([9.,-3.,20.])
        z,y,x = np.meshgrid([0.,2.,4.],[0.,1.,2.,3.],[0.,1.,2.,3.,4.],indexing='ij')
        xyz = origin + np.stack([x,y,z],axis=-1) @ basis.T
        data = 2*xyz[...,0]+3*xyz[...,1]+5*xyz[...,2]
        volume = _Volume(data,origin,basis,1.,1.,np.array([0.,2.,4.]),{})
        point = origin + np.array([1.3,2.2,2.7]) @ basis.T
        self.assertAlmostEqual(float(volume.sample(point).item()), 2*point[0]+3*point[1]+5*point[2], places=7)

    def test_absolute_and_descending_frame_offsets(self):
        original, _ = _read_dose(self.dose_path)
        ds = pydicom.dcmread(self.dose_path)
        ds.GridFrameOffsetVector = [10,11,14]
        self.write_dicom(ds)
        absolute, _ = _read_dose(self.dose_path)
        point = [2.,-3.,12.]
        self.assertAlmostEqual(float(original.sample(point).item()),float(absolute.sample(point).item()))
        ds.ImagePositionPatient = [0,-5,14]
        ds.GridFrameOffsetVector = [0,-3,-4]
        ds.PixelData = ds.pixel_array[::-1].copy().tobytes()
        self.write_dicom(ds)
        descending, _ = _read_dose(self.dose_path)
        self.assertAlmostEqual(float(original.sample(point).item()),float(descending.sample(point).item()))
        ds.GridFrameOffsetVector = [14,11,10]
        self.write_dicom(ds)
        absolute_descending, _ = _read_dose(self.dose_path)
        self.assertAlmostEqual(float(original.sample(point).item()),
                               float(absolute_descending.sample(point).item()))

    def test_identity_mismatches_are_rejected_before_output(self):
        original = pydicom.dcmread(self.dose_path)
        output = self.root/'must_not_exist.png'
        for field in ('PatientID', 'StudyInstanceUID', 'FrameOfReferenceUID'):
            for value in ('', generate_uid()):
                with self.subTest(field=field, value=value):
                    ds = deepcopy(original)
                    setattr(ds, field, value)
                    self.write_dicom(ds)
                    with self.assertRaisesRegex(ValueError, field):
                        self.render(output)
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_suffix('.geometry.json').exists())

    def test_multiple_ct_series_require_explicit_selection(self):
        ds = pydicom.dcmread(self.ct_path/'0.dcm')
        ds.SeriesInstanceUID, ds.SOPInstanceUID = generate_uid(), generate_uid()
        self.write_dicom(ds, self.ct_path/'extra.dcm')
        dose, _ = _read_dose(self.dose_path)
        with self.assertRaisesRegex(ValueError,'ct_series_uid'):
            _read_ct(self.ct_path,dose,None,1)
        ct, _ = _read_ct(self.ct_path,dose,self.ct_series,1)
        self.assertEqual(ct.data.shape,(3,4,5))

    def test_masking_and_rendered_report(self):
        dose, _ = _read_dose(self.dose_path)
        self.assertTrue(np.isnan(dose.sample([100.,100.,100.])).all())
        sources = {path: path.read_bytes() for path in [self.dose_path, *self.ct_path.glob('*.dcm')]}
        output = self.root/'nested'/'overlay.png'
        report = self.render(output, center_xyz_mm=[2,-4,12])
        output.write_bytes(b'old figure')
        output.with_suffix('.geometry.json').write_text('old report')
        report = self.render(output, center_xyz_mm=[2,-4,12])
        self.assertEqual(output.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        persisted = json.loads(output.with_suffix('.geometry.json').read_text())
        self.assertEqual(report,persisted)
        self.assertAlmostEqual(report['dose_max'],42)
        self.assertEqual(report['dose']['DoseType'],'EFFECTIVE')
        self.assertAlmostEqual(report['display']['dose_threshold'],4.2)
        self.assertFalse(report['checks']['plan_reference_checked'])
        self.assertEqual(report['display']['center_xyz_mm'],[2.,-4.,12.])
        for path, data in sources.items():
            self.assertEqual(path.read_bytes(), data)

    def test_invalid_display_and_outside_center_are_rejected(self):
        cases = [
            ({'dose_threshold_fraction': 10}, 'Threshold fractions'),
            ({'center_threshold_fraction': 0}, 'Threshold fractions'),
            ({'ct_ww': 0}, 'ct_ww'), ({'ct_wl': np.nan}, 'finite'),
            ({'alpha': 0}, 'alpha'), ({'workers': True}, 'positive integers'),
            ({'workers': 0}, 'positive integers'), ({'dpi': 1.5}, 'positive integers'),
            ({'dpi': False}, 'positive integers'), ({'cmap': 'not-a-cmap'}, 'cmap'),
            ({'pick': 'invalid'}, 'pick'),
            ({'center_xyz_mm': [99,99,99]}, 'outside the CT'),
            ({'center_xyz_mm': [2,3]}, 'three finite'),
            ({'center_xyz_mm': [2,np.nan,12]}, 'three finite'),
        ]
        for options, message in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    self.render(**options)
                self.assertFalse((self.root/'overlay.png').exists())
                self.assertFalse((self.root/'overlay.geometry.json').exists())

    def test_invalid_ct_geometry_is_rejected(self):
        path = self.ct_path/'0.dcm'
        original = pydicom.dcmread(path)
        cases = [
            ('ImagePositionPatient', [4,-2,12], 'duplicate'),
            ('ImagePositionPatient', [4,-2,16], 'missing slices'),
            ('ImagePositionPatient', [5,-2,14], 'in-plane shifts'),
            ('ImagePositionPatient', [4,-2], 'three finite'),
            ('ImageOrientationPatient', [1,0,0,0,1,0], 'orientations differ'),
            ('PixelSpacing', [1,2], 'spacing differs'),
            ('PixelSpacing', [-1,1], 'positive finite'),
            ('Rows', 3, 'sizes differ'),
            ('SOPInstanceUID', pydicom.dcmread(self.ct_path/'1.dcm').SOPInstanceUID, 'duplicate SOP'),
            ('SOPInstanceUID', '', 'SOPInstanceUID'),
            ('NumberOfFrames', 2, 'single-frame CT'),
            ('RescaleType', 'OTHER', 'HU'),
            ('RescaleSlope', float('inf'), 'finite'),
        ]
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                ds = deepcopy(original)
                setattr(ds, field, value)
                self.write_dicom(ds, path)
                with self.assertRaisesRegex(ValueError, message):
                    self.render()
                self.assertFalse((self.root/'overlay.png').exists())
        for field in ('RescaleSlope', 'RescaleIntercept'):
            ds = deepcopy(original)
            del ds[field]
            self.write_dicom(ds, path)
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, 'HU conversion'):
                self.render()

    def test_invalid_dose_geometry_and_values_are_rejected(self):
        original = pydicom.dcmread(self.dose_path)
        cases = [
            ('GridFrameOffsetVector', [0,1], 'one finite offset'),
            ('GridFrameOffsetVector', [0,1,float('nan')], 'one finite offset'),
            ('GridFrameOffsetVector', [0,2,1], 'strictly increasing'),
            ('GridFrameOffsetVector', [0,1,1], 'strictly increasing'),
            ('GridFrameOffsetVector', [1,2,3], 'relative/absolute'),
            ('ImageOrientationPatient', [1,0,0,0,2,0], 'orthonormal'),
            ('ImageOrientationPatient', [1,0,0], 'six values'),
            ('DoseGridScaling', 0, 'positive and finite'),
            ('DoseUnits', 'UNKNOWN', 'DoseUnits'),
            ('PixelData', np.zeros_like(original.pixel_array).tobytes(), 'some positive dose'),
        ]
        for field, value, message in cases:
            with self.subTest(field=field):
                ds = deepcopy(original)
                setattr(ds, field, value)
                self.write_dicom(ds)
                with self.assertRaisesRegex(ValueError, message):
                    self.render()
        original.ImagePositionPatient = [0,-5,10000]
        original.GridFrameOffsetVector = [10000.05,10001,10004]
        self.write_dicom(original)
        with self.assertRaisesRegex(ValueError, 'relative/absolute'):
            self.render()

    def test_oblique_ct_and_dose_are_sampled_on_patient_planes(self):
        angle = np.pi/6
        rotation = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],
                             [0,np.sin(angle),np.cos(angle)]])
        for path in [self.dose_path, *self.ct_path.glob('*.dcm')]:
            ds = pydicom.dcmread(path)
            basis = rotation @ _basis(ds.ImageOrientationPatient)
            ds.ImageOrientationPatient = np.concatenate((basis[:,0], basis[:,1])).tolist()
            ds.ImagePositionPatient = (rotation @ ds.ImagePositionPatient).tolist()
            self.write_dicom(ds, path)
        dose, _ = _read_dose(self.dose_path)
        ct, _ = _read_ct(self.ct_path, dose, None, 1)
        point = rotation @ [2.5,-3.5,12.5]
        self.assertAlmostEqual(dose.sample(point).item(), 33, places=5)
        self.assertAlmostEqual(ct.sample(point).item(), 232, places=5)
        panels = _planes(ct, dose, point, None)
        self.assertTrue(any(np.isnan(panel['ct']).any() for panel in panels))
        for panel in panels:
            self.assertTrue(np.isnan(panel['dose'][~np.isfinite(panel['ct'])]).all())
            self.assertTrue(np.isfinite(panel['dose']).any())

    def test_center_methods_and_explicit_override(self):
        dose, _ = _read_dose(self.dose_path)
        ct, _ = _read_ct(self.ct_path, dose, None, 1)
        center, _ = _center(dose, ct, 'center', None, .5)
        np.testing.assert_allclose(center, [2,-3.5,12])
        center, _ = _center(dose, ct, 'maxdose', None, .5)
        np.testing.assert_allclose(center, [4,-2,14])
        center, _ = _center(dose, ct, 'dose_center', None, .75)
        z,y,x = np.meshgrid([10,11,14], [-5,-4,-3,-2], [0,1,2,3,4], indexing='ij')
        weights = dose.data * (dose.data >= .75 * 42)
        expected = [np.sum(axis * weights)/weights.sum() for axis in (x,y,z)]
        np.testing.assert_allclose(center, expected)
        center, _ = _center(dose, ct, 'maxdose', [1,-3,11], .5)
        np.testing.assert_allclose(center, [1,-3,11])
        ds = pydicom.dcmread(self.dose_path)
        ds.ImagePositionPatient = [100,-5,10]
        self.write_dicom(ds)
        with self.assertRaisesRegex(ValueError, 'no positive dose'):
            self.render(pick='center')

    def test_crop_validation_and_numpy_report_values(self):
        cases = [
            ({'x': [99,100]}, 'does not overlap'),
            ({'x': [0,1]}, 'excludes'),
            ({'x': [3,1]}, 'minimum, maximum'),
            ({'x': [np.nan,3]}, 'finite'),
            ({'z': [12]}, 'pairs'),
            ({'other': [0,1]}, 'keys'),
            ('x', 'must map'),
        ]
        for crop, message in cases:
            with self.subTest(crop=crop):
                with self.assertRaisesRegex(ValueError, message):
                    self.render(center_xyz_mm=[2,-4,12], crop_mm=crop)
        report = self.render(center_xyz_mm=[2,-4,12], crop_mm={'x': np.array([1,3])},
                             alpha=np.float32(.5), dose_threshold_fraction=np.float32(.25))
        self.assertEqual(report['display']['crop_mm'], {'x': [1.,3.]})
        self.assertEqual(report['display']['alpha'], .5)
        self.assertEqual(report, json.loads((self.root/'overlay.geometry.json').read_text()))

    def test_panel_orientation_scale_crosshairs_and_units(self):
        ds = pydicom.dcmread(self.dose_path)
        ds.DoseUnits = 'RELATIVE'
        self.write_dicom(ds)
        original_savefig = Figure.savefig
        inspected = []

        def inspect_figure(fig, *args, **kwargs):
            panels, colorbar = fig.axes[:3], fig.axes[3]
            for ax, name, labels, point in zip(
                    panels, ('Axial', 'Sagittal', 'Coronal'),
                    (('R','L','A','P'), ('A','P','S','I'), ('R','L','S','I')),
                    ((2,-4), (-4,12), (2,12))):
                self.assertIn(name, ax.get_title(loc='left'))
                self.assertEqual(tuple(text.get_text() for text in ax.texts), labels)
                self.assertEqual(ax.get_aspect(), 1)
                self.assertEqual(ax.yaxis_inverted(), name == 'Axial')
                self.assertEqual(ax.images[0].get_clim(), (-160,240))
                self.assertEqual(ax.images[1].get_clim(), (0,42))
                overlay = ax.images[1].get_array()
                self.assertLess(overlay.max(), 42)
                self.assertTrue(np.all(overlay.compressed() >= 33.6))
                np.testing.assert_allclose(ax.lines[0].get_xdata(), point[0])
                np.testing.assert_allclose(ax.lines[1].get_ydata(), point[1])
            self.assertEqual(colorbar.get_ylabel(), 'Dose (relative units; DICOM EFFECTIVE)')
            np.testing.assert_allclose(colorbar.get_yticks(), np.linspace(0,42,6))
            inspected.append(True)
            return original_savefig(fig, *args, **kwargs)

        import matplotlib
        backend = matplotlib.get_backend()
        with patch.object(Figure, 'savefig', autospec=True, side_effect=inspect_figure):
            self.render(center_xyz_mm=[2,-4,12], dose_threshold_fraction=.8)
        self.assertEqual(inspected, [True])
        self.assertEqual(matplotlib.get_backend(), backend)

    def test_reference_chain_and_class_mismatch_note(self):
        self.reference_files()
        ds = pydicom.dcmread(self.dose_path)
        ds.ReferencedRTPlanSequence[0].ReferencedSOPClassUID = RTStructureSetStorage
        self.write_dicom(ds)
        report = self.render(rtplan_path=self.plan_path, rtstruct_path=self.struct_path)
        for key in ('dose_to_plan_instance_uid_match', 'plan_to_structure_instance_uid_match',
                    'structure_to_ct_series_match', 'referenced_ct_images_checked',
                    'plan_reference_checked', 'structure_reference_checked'):
            self.assertTrue(report['checks'][key])
        self.assertEqual(report['checks']['referenced_ct_images_present'], 3)
        self.assertIn('ReferencedSOPClassUID', report['reference_metadata_notes'][0])

    def test_partial_reference_checks_are_reported(self):
        _, struct = self.reference_files()
        plan_only = self.render(rtplan_path=self.plan_path)
        self.assertTrue(plan_only['checks']['plan_reference_checked'])
        self.assertFalse(plan_only['checks']['structure_reference_checked'])
        self.assertFalse(plan_only['checks']['plan_to_structure_reference_checked'])
        series = struct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
        del series.ContourImageSequence
        self.write_dicom(struct, self.struct_path)
        structure_only = self.render(rtstruct_path=self.struct_path)
        self.assertFalse(structure_only['checks']['plan_reference_checked'])
        self.assertTrue(structure_only['checks']['structure_reference_checked'])
        self.assertFalse(structure_only['checks']['plan_to_structure_reference_checked'])
        self.assertFalse(structure_only['checks']['referenced_ct_images_checked'])
        self.assertIn('slice availability was not checked', structure_only['reference_metadata_notes'][0])

    def test_reference_mismatches_are_rejected(self):
        for broken_link, message in (
                ('dose-plan', 'does not reference the supplied RTPLAN'),
                ('plan-structure', 'does not reference the supplied RTSTRUCT'),
                ('structure-series', 'does not reference the selected CT series'),
                ('structure-images', 'CT slices are missing'),
                ('structure-frame', 'does not reference the CT/dose frame')):
            with self.subTest(link=broken_link):
                plan, struct = self.reference_files()
                frame = struct.ReferencedFrameOfReferenceSequence[0]
                series = frame.RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
                if broken_link == 'dose-plan':
                    ds = pydicom.dcmread(self.dose_path)
                    ds.ReferencedRTPlanSequence = []
                    self.write_dicom(ds)
                elif broken_link == 'plan-structure':
                    plan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = generate_uid()
                elif broken_link == 'structure-series':
                    series.SeriesInstanceUID = generate_uid()
                elif broken_link == 'structure-images':
                    series.ContourImageSequence[0].ReferencedSOPInstanceUID = generate_uid()
                else:
                    frame.FrameOfReferenceUID = generate_uid()
                self.write_dicom(plan, self.plan_path)
                self.write_dicom(struct, self.struct_path)
                with self.assertRaisesRegex(ValueError, message):
                    self.render(rtplan_path=self.plan_path, rtstruct_path=self.struct_path)
                self.assertFalse((self.root/'overlay.png').exists())

    def test_imports_cli_listing_and_headless_render(self):
        import dose_view
        from dose_view import triplanar, visualization
        self.assertIs(dose_view.save_triplanar_overlay, triplanar.save_triplanar_overlay)
        self.assertIs(dose_view.save_triplanar_overlays, visualization.save_triplanar_overlays)
        env = dict(os.environ, MPLBACKEND='Agg')
        env.pop('DISPLAY', None)
        command = [sys.executable, '-m', 'dose_view.cli']
        listing = subprocess.run(command + ['list'], env=env, check=True, capture_output=True, text=True)
        self.assertIn('save_triplanar_overlay(', listing.stdout)
        output = self.root/'cli.png'
        kwargs = dict(ct_path=str(self.ct_path), rtdose_path=str(self.dose_path),
                      output_path=str(output), workers=1, dpi=40)
        result = subprocess.run(command + ['triplanar.save_triplanar_overlay', '--kwargs', json.dumps(kwargs)],
                                env=env, check=True, capture_output=True, text=True)
        self.assertIn(str(output), result.stdout)
        self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(json.loads(output.with_suffix('.geometry.json').read_text())['dose_max'], 42)

    def test_missing_extras_and_unrelated_import_errors(self):
        original_import = builtins.__import__
        for missing in ('pydicom', 'scipy', 'unrelated_dependency'):
            def missing_import(name, *args, **kwargs):
                if name == ('pydicom' if missing == 'pydicom' else 'scipy.interpolate'):
                    raise ModuleNotFoundError(f'No module named {missing}', name=missing)
                return original_import(name, *args, **kwargs)
            with self.subTest(missing=missing), patch('builtins.__import__', side_effect=missing_import):
                message = 'dose_view\\[dicom\\]' if missing != 'unrelated_dependency' else missing
                with self.assertRaisesRegex(ImportError, message):
                    self.render()

    def test_input_and_output_errors_are_not_hidden(self):
        with self.assertRaisesRegex(ValueError, 'end in .png'):
            self.render(self.root/'bad.pdf')
        with patch('dose_view.triplanar._render', side_effect=OSError('write blocked')):
            with self.assertRaisesRegex(OSError, 'write blocked'):
                self.render()
        self.assertFalse((self.root/'overlay.geometry.json').exists())
        self.reference_files()
        with self.assertRaises(FileNotFoundError):
            self.render(rtplan_path=self.root/'missing.dcm')
        self.assertFalse((self.root/'overlay.png').exists())

    def test_output_paths_cannot_overwrite_dicom_inputs(self):
        original = self.dose_path.read_bytes()
        output = self.root/'overlay.png'
        sidecar = output.with_suffix('.geometry.json')
        sidecar.symlink_to(self.dose_path)
        with self.assertRaisesRegex(ValueError, 'overwrite DICOM'):
            self.render(output)
        self.assertEqual(self.dose_path.read_bytes(), original)
        self.assertFalse(output.exists())
        sidecar.unlink()
        self.dose_path = self.root/'input.png'
        self.dose_path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'overwrite DICOM'):
            self.render(self.dose_path)
        self.assertEqual(self.dose_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
