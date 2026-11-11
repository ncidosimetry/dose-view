"""Analytical, synthetic DICOM tests; no patient data or mounted volumes."""
from copy import deepcopy
import json
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

import test_triplanar as fixtures
from dose_view import calculate_dvh, save_dvh, list_rois
from dose_view.dvh import _inside_polygon


class DVHTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TriplanarTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.plan, self.struct = self.fixture.reference_files()
        definition = Dataset()
        definition.ROINumber, definition.ROIName = 1, 'Synthetic target'
        definition.ReferencedFrameOfReferenceUID = self.fixture.frame
        self.struct.StructureSetROISequence = [definition]
        roi = Dataset()
        roi.ReferencedROINumber = 1
        roi.ContourSequence = [self.rectangle(.5, 3.5, -4.5, -2.5)]
        self.struct.ROIContourSequence = [roi]
        self.write_struct()

    def rectangle(self, x0, x1, y0, y1, z=12, kind='CLOSED_PLANAR'):
        return self.contour([[x0,y0,z], [x1,y0,z], [x1,y1,z], [x0,y1,z]], kind)

    def contour(self, points, kind='CLOSED_PLANAR'):
        c = Dataset()
        c.ContourGeometricType = kind
        c.NumberOfContourPoints = len(points)
        c.ContourData = np.asarray(points, dtype=float).ravel().tolist()
        path = self.fixture.ct_path / f'{2-int((points[0][2]-10)/2)}.dcm'
        c.ContourImageSequence = [self.fixture.reference(pydicom.dcmread(path))]
        return c

    def write_struct(self):
        self.fixture.write_dicom(self.struct, self.fixture.struct_path)

    def calculate(self, **kwargs):
        options = dict(roi_numbers=[1], rtplan_path=self.fixture.plan_path, workers=1)
        options.update(kwargs)
        return calculate_dvh(self.fixture.ct_path, self.fixture.dose_path, self.fixture.struct_path, **options)

    def set_uniform(self, gy):
        ds = pydicom.dcmread(self.fixture.dose_path)
        ds.PixelData = np.full(ds.pixel_array.shape, round(gy/float(ds.DoseGridScaling)), dtype='<i2').tobytes()
        self.fixture.write_dicom(ds)

    def test_uniform_dose_volume_metrics_and_exact_threshold(self):
        self.set_uniform(10)
        report = self.calculate(v_doses_gy=[10, 10.1], bin_width_gy=1)
        roi = report['rois'][0]
        self.assertAlmostEqual(roi['sampled_volume_cc'], .012)
        self.assertEqual(roi['coverage_pct'], 100)
        self.assertEqual(roi['sample_count'], 24)
        for key in ('Dmin_Gy','Dmax_Gy','Dmean_Gy','D2_Gy','D50_Gy','D95_Gy','D98_Gy'):
            self.assertAlmostEqual(roi['metrics'][key], 10)
        self.assertEqual([v['volume_pct'] for v in roi['metrics']['V_dose']], [100, 0])
        curve = roi['cumulative']
        self.assertAlmostEqual(curve['volume_pct'][10], 100)
        self.assertEqual(curve['volume_pct'][11], 0)
        self.assertTrue(report['checks']['plan_to_structure_reference_checked'])
        json.dumps(report, allow_nan=False)

    def test_gradient_uses_patient_coordinates_and_nonuniform_dose_planes(self):
        roi = self.calculate(v_doses_gy=[31])['rois'][0]
        self.assertAlmostEqual(roi['metrics']['Dmean_Gy'], 31)
        self.assertAlmostEqual(roi['metrics']['Dmin_Gy'], 28.25)
        self.assertAlmostEqual(roi['metrics']['Dmax_Gy'], 33.75)
        self.assertEqual(roi['metrics']['V_dose'][0]['volume_pct'], 50)
        # Equal-volume analytical samples, independent of histogram bin width.
        values = np.sort([20+x+2*(y+5)+6 for x in np.arange(.75,3.5,.5)
                          for y in np.arange(-4.25,-2.5,.5)])
        self.assertAlmostEqual(roi['metrics']['D95_Gy'], values[1])
        self.assertAlmostEqual(roi['metrics']['D98_Gy'], values[0])
        other = self.calculate(bin_width_gy=5)['rois'][0]
        self.assertEqual(roi['metrics']['D95_Gy'], other['metrics']['D95_Gy'])

    def test_zero_dose_is_valid(self):
        self.set_uniform(0)
        roi = self.calculate()['rois'][0]
        self.assertEqual(roi['metrics']['Dmean_Gy'], 0)
        self.assertEqual(roi['cumulative']['volume_pct'], [100, 0])

    def test_closed_planar_union_xor_hole_and_keyhole(self):
        roi = self.struct.ROIContourSequence[0]
        outer = self.rectangle(.5,3.5,-4.5,-2.5)
        inner = self.rectangle(1.5,2.5,-4,-3)
        roi.ContourSequence = [outer, inner]
        self.write_struct()
        self.assertAlmostEqual(self.calculate()['rois'][0]['sampled_volume_cc'], .012)
        for c in roi.ContourSequence:
            c.ContourGeometricType = 'CLOSEDPLANAR_XOR'
        self.write_struct()
        self.assertAlmostEqual(self.calculate()['rois'][0]['sampled_volume_cc'], .010)
        a = [.5,-4.5,12]
        h = [1.5,-4,12]
        roi.ContourSequence = [self.contour([a,[3.5,-4.5,12],[3.5,-2.5,12],[.5,-2.5,12],a,
                                            h,[1.5,-3,12],[2.5,-3,12],[2.5,-4,12],h])]
        self.write_struct()
        self.assertAlmostEqual(self.calculate()['rois'][0]['sampled_volume_cc'], .010)
        p = np.array([[0,0],[2,0],[2,2],[0,2]])
        np.testing.assert_array_equal(_inside_polygon(np.array([[0,1],[1,1],[3,1]]),p),[True,True,False])

    def test_oblique_patient_geometry_preserves_analytical_result(self):
        baseline = self.calculate()['rois'][0]
        angle = np.pi/6
        rot = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        for path in [self.fixture.dose_path, *self.fixture.ct_path.glob('*.dcm')]:
            ds = pydicom.dcmread(path)
            ds.ImagePositionPatient = (rot @ np.asarray(ds.ImagePositionPatient)).tolist()
            directions = np.asarray(ds.ImageOrientationPatient).reshape(2,3) @ rot.T
            ds.ImageOrientationPatient = directions.ravel().tolist()
            self.fixture.write_dicom(ds, path)
        for c in self.struct.ROIContourSequence[0].ContourSequence:
            c.ContourData = (np.asarray(c.ContourData).reshape(-1,3) @ rot.T).ravel().tolist()
        self.write_struct()
        result = self.calculate()['rois'][0]
        self.assertAlmostEqual(result['sampled_volume_cc'], baseline['sampled_volume_cc'])
        for field in ('Dmean_Gy','D95_Gy','D98_Gy'):
            self.assertAlmostEqual(result['metrics'][field], baseline['metrics'][field], places=5)

    def test_partial_and_absent_coverage_never_become_whole_roi_metrics(self):
        self.struct.ROIContourSequence[0].ContourSequence = [self.rectangle(-.4,3.5,-4.5,-2.5)]
        self.write_struct()
        with self.assertRaisesRegex(ValueError, 'outside RTDOSE'):
            self.calculate()
        roi = self.calculate(allow_partial=True)['rois'][0]
        self.assertEqual(roi['coverage'], 'partial')
        self.assertIsNone(roi['metrics'])
        self.assertLess(roi['coverage_pct'], 100)
        self.assertGreater(roi['uncovered_volume_cc'], 0)
        self.assertIsNotNone(roi['covered_metrics'])
        ds = pydicom.dcmread(self.fixture.dose_path)
        ds.ImagePositionPatient = [100,-5,10]
        self.fixture.write_dicom(ds)
        roi = self.calculate(allow_partial=True)['rois'][0]
        self.assertEqual(roi['coverage_pct'], 0)
        self.assertIsNone(roi['covered_metrics'])
        self.assertIsNone(roi['cumulative']['volume_pct'])

    def test_invalid_geometry_and_contour_semantics_fail(self):
        original = self.fixture.struct_path.read_bytes()
        cases = [
            ('ContourGeometricType','OPEN_PLANAR','only unmixed'),
            ('NumberOfContourPoints',3,'point count'),
            ('ContourSlabThickness',2,'slabs/offsets'),
            ('ContourData',[.5,-4.5,12,3.5,-4.5,12,3.5,-2.5,12.1,.5,-2.5,12],'nonplanar'),
            ('ContourData',[-1,-4.5,12,3.5,-4.5,12,3.5,-2.5,12,-1,-2.5,12],'outside the CT'),
        ]
        for field, value, message in cases:
            with self.subTest(field=field):
                self.struct = pydicom.dcmread(BytesIO(original))
                setattr(self.struct.ROIContourSequence[0].ContourSequence[0],field,value)
                self.write_struct()
                with self.assertRaisesRegex(ValueError,message):
                    self.calculate()
        self.struct = pydicom.dcmread(BytesIO(original))
        self.struct.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID = generate_uid()
        self.write_struct()
        with self.assertRaisesRegex(ValueError, 'image reference'):
            self.calculate()
        self.struct = pydicom.dcmread(BytesIO(original))
        self.struct.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = generate_uid()
        self.write_struct()
        with self.assertRaisesRegex(ValueError, 'ReferencedFrameOfReferenceUID'):
            self.calculate()

    def test_explicit_selection_identity_units_and_options(self):
        self.assertEqual(list_rois(self.fixture.struct_path)[0]['name'], 'Synthetic target')
        with self.assertRaisesRegex(ValueError, 'Select roi_numbers'):
            self.calculate(roi_numbers=None)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.calculate(roi_numbers=[99])
        self.assertEqual(self.calculate(roi_numbers=None,roi_names=['Synthetic target'])['rois'][0]['number'],1)
        duplicate = deepcopy(self.struct.StructureSetROISequence[0])
        duplicate.ROINumber = 2
        self.struct.StructureSetROISequence.append(duplicate)
        self.write_struct()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.calculate(roi_names=['Synthetic target'])
        for opts in ({'in_plane_subsamples':0},{'in_plane_subsamples':True},{'bin_width_gy':0},
                     {'v_doses_gy':[float('nan')]},{'workers':0}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                self.calculate(**opts)
        ds = pydicom.dcmread(self.fixture.dose_path)
        ds.DoseUnits = 'RELATIVE'
        self.fixture.write_dicom(ds)
        with self.assertRaisesRegex(ValueError, 'DoseUnits GY'):
            self.calculate()
        ds.DoseUnits = 'GY'
        ds.PatientID = 'OTHER'
        self.fixture.write_dicom(ds)
        with self.assertRaisesRegex(ValueError, 'PatientID'):
            self.calculate()

    def test_gaps_and_missing_contour_references_are_explicit(self):
        roi = self.struct.ROIContourSequence[0]
        roi.ContourSequence = [self.rectangle(.5,3.5,-4.5,-2.5,z=z) for z in (10,14)]
        for c in roi.ContourSequence:
            del c.ContourImageSequence
        self.write_struct()
        result = self.calculate()['rois'][0]
        self.assertAlmostEqual(result['sampled_volume_cc'], .024)
        self.assertEqual(result['geometry']['uncontoured_planes_between_extents'], [1])
        self.assertEqual(len(result['warnings']), 2)

    def test_outputs_cli_and_source_preservation(self):
        sources = {p:p.read_bytes() for p in [self.fixture.struct_path,self.fixture.plan_path,
                                             self.fixture.dose_path,*self.fixture.ct_path.glob('*.dcm')]}
        output = self.root/'out'/'dvh.png'
        report = save_dvh(self.fixture.ct_path,self.fixture.dose_path,self.fixture.struct_path,
                          output,roi_numbers=[1],rtplan_path=self.fixture.plan_path,workers=1,dpi=40)
        self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(json.loads(output.with_suffix('.dvh.json').read_text()),report)
        self.assertIn('volume_pct_of_covered',output.with_suffix('.dvh.csv').read_text())
        with self.assertRaises(FileExistsError):
            save_dvh(self.fixture.ct_path,self.fixture.dose_path,self.fixture.struct_path,
                     output,roi_numbers=[1])
        for path, content in sources.items():
            self.assertEqual(path.read_bytes(),content)
        command = [sys.executable,'-m','dose_view.cli','dvh','--ct',str(self.fixture.ct_path),
                   '--dose',str(self.fixture.dose_path),'--struct',str(self.fixture.struct_path),
                   '--plan',str(self.fixture.plan_path),'--roi-number','1',
                   '--output',str(self.root/'cli.png'),'--v-dose','31']
        result = subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(Path(json.loads(result.stdout)['outputs']['figure']).exists())
        command[command.index('--roi-number')+1] = '999'
        command[command.index('--output')+1] = str(self.root/'bad.png')
        result = subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(result.returncode,2)
        self.assertFalse((self.root/'bad.png').exists())


if __name__ == '__main__':
    unittest.main()
