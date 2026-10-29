"""Контракт экспорта: утечки, численная совместимость и целостность выпуска."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.mobile_export import (MobileModel, calibration_rows, calibrate, softmax,
                               quality_gate, package, verify_release, synthetic_inputs)
from src.small_classifier import transform


class MobileExportTests(unittest.TestCase):
    def test_argmax_policy_removes_only_confidence_check(self):
        from src.mobile_export import target_matches
        mapping = {'arch': 0, 'tower': 1}
        policy = {'mode': 'argmax_only', 'confidence_rejection_enabled': False, 'threshold': 0.0}
        self.assertTrue(target_matches([.001, 0], 'arch', mapping, policy))
        self.assertFalse(target_matches([.001, 0], 'tower', mapping, policy))
        self.assertFalse(target_matches([.001, 0], 'unknown', mapping, policy))
        self.assertFalse(target_matches([np.nan, 0], 'arch', mapping, policy))
        self.assertFalse(target_matches([1], 'arch', mapping, policy))
        self.assertFalse(target_matches([.001, 0], 'arch', mapping, {'threshold': .95}))
    def test_failed_int8_does_not_block_explicit_validated_fp32_or_enter_index(self):
        from scripts.build_mobile_release_index import build_index
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / 'public/v1/fp32'
            public.mkdir(parents=True)
            (public / 'manifest.json').write_text('{}')
            evaluation = {'validation_gate_passed': True}
            (root / 'status.json').write_text(json.dumps({'status': 'failed', 'completed_variants': ['fp32']}))
            (root / 'comparison.json').write_text(json.dumps({'release_version': 'v1', 'variants': {'fp32': evaluation}}))
            (root / 'fp32_validation_gate.json').write_text(json.dumps({'passed': True}))
            verified = {'evaluation': evaluation, 'files': {'model.tflite': {'bytes': 100}}}
            with patch('scripts.build_mobile_release_index.verify_release', return_value=verified):
                with self.assertRaises(ValueError):
                    build_index(root)
                index = build_index(root, ['fp32'])
                self.assertEqual(set(index['variants']), {'fp32'})
                self.assertEqual(index['excluded_variants'], ['int8'])
                self.assertFalse(index['automatic_activation'])
                with self.assertRaises(ValueError):
                    build_index(root, ['int8'])
                (root / 'fp32_validation_gate.json').write_text(json.dumps({'passed': False}))
                with self.assertRaises(ValueError):
                    build_index(root, ['fp32'])

    def test_infer_handles_integer_io_end_to_end(self):
        from scripts.export_mobile_litert import infer
        class IntegerInterpreter:
            _nevidal_runtime = {'backend': 'builtin_without_default_delegates'}
            def get_input_details(self):
                return [dict(index=4, name='rgb', dtype=np.int8, shape=np.array([1,224,224,3]), quantization=(1., -128))]
            def get_output_details(self):
                return [dict(index=8, name='logits', dtype=np.int8, shape=np.array([1,23]), quantization=(.5, -2))]
            def set_tensor(self, index, value):
                assert index == 4 and value.dtype == np.int8
                np.testing.assert_array_equal(value[0,0,0], [127,-128,-128])
            def invoke(self):
                pass
            def get_tensor(self, index):
                assert index == 8
                return np.arange(-2,21, dtype=np.int8)[None]
            def get_tensor_details(self):
                return self.get_input_details() + self.get_output_details()
        with patch('scripts.export_mobile_litert.interpreter_for', return_value=IntegerInterpreter()):
            logits, contract = infer(Path('int8.tflite'), synthetic_inputs()[:1])
        np.testing.assert_array_equal(logits, np.arange(23)[None] * .5)
        self.assertEqual(contract['input']['zero_point'], -128)
        self.assertEqual(contract['output']['scale'], .5)

    def test_integer_input_scaling_clipping_and_tie_rounding(self):
        from scripts.export_mobile_litert import encode_input
        detail = dict(dtype=np.int8, quantization=(2., -128))
        np.testing.assert_array_equal(encode_input([0, 1, 2, 255, 999], detail), [-128, -128, -127, -1, 127])
        # Половины от нуля: -127.5 -> -128, -0.5 -> -1.
        unsigned = dict(dtype=np.uint8, quantization=(1., 0))
        np.testing.assert_array_equal(encode_input([0, .5, 255, 300], unsigned), [0, 1, 255, 255])

    def test_integer_output_dequantized_before_softmax(self):
        from scripts.export_mobile_litert import decode_output
        detail = dict(dtype=np.int8, quantization=(.25, -3))
        logits = decode_output(np.array([[-3, 1, -7]], dtype=np.int8), detail)
        np.testing.assert_array_equal(logits, [[0., 1., -1.]])
        np.testing.assert_allclose(softmax(logits), softmax([[0., 1., -1.]]))

    def test_invalid_quantization_fails_explicitly(self):
        from scripts.export_mobile_litert import validate_io
        for detail in (dict(dtype=np.int8, quantization=(0, 0)),
                       dict(dtype=np.int16, quantization=(1, 0)),
                       dict(dtype=np.uint8, quantization=(1, -1)),
                       dict(dtype=np.int8, quantization=(1, 0), quantization_parameters={'scales': [1, 2]})):
            with self.assertRaises(ValueError):
                validate_io(detail)

    def test_float_io_is_unchanged(self):
        from scripts.export_mobile_litert import encode_input, decode_output
        detail = dict(dtype=np.float32)
        values = np.array([[0, 124, 255]], dtype=np.float32)
        np.testing.assert_array_equal(encode_input(values, detail), values)
        np.testing.assert_array_equal(decode_output(values, detail), values)

    def test_xnnpack_failure_recreates_interpreter_without_delegates(self):
        from scripts.export_mobile_litert import interpreter_for
        calls = []
        class FakeInterpreter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                calls.append(kwargs)
            def allocate_tensors(self):
                if 'experimental_op_resolver_type' not in self.kwargs:
                    raise RuntimeError('failed to create XNNPACK runtime')
            def get_input_details(self):
                return [dict(shape=np.array([1, 224, 224, 3]), dtype=np.float32)]
            def get_output_details(self):
                return [dict(shape=np.array([1, 23]), dtype=np.float32)]
        module = types.ModuleType('ai_edge_litert.interpreter')
        module.Interpreter = FakeInterpreter
        module.OpResolverType = types.SimpleNamespace(BUILTIN_WITHOUT_DEFAULT_DELEGATES=3)
        with patch.dict('sys.modules', {'ai_edge_litert.interpreter': module}):
            result = interpreter_for(Path('model_int8.tflite'))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]['experimental_op_resolver_type'], 3)
        self.assertEqual(result._nevidal_runtime['backend'], 'builtin_without_default_delegates')

    def test_other_runtime_errors_are_not_hidden(self):
        from scripts.export_mobile_litert import interpreter_for
        class BadInterpreter:
            def __init__(self, **kwargs):
                pass
            def allocate_tensors(self):
                raise RuntimeError('corrupt model')
        module = types.ModuleType('ai_edge_litert.interpreter')
        module.Interpreter = BadInterpreter
        module.OpResolverType = types.SimpleNamespace(BUILTIN_WITHOUT_DEFAULT_DELEGATES=3)
        with patch.dict('sys.modules', {'ai_edge_litert.interpreter': module}):
            with self.assertRaisesRegex(RuntimeError, 'corrupt model'):
                interpreter_for(Path('bad.tflite'))

    def test_preprocessing_matches_training(self):
        image = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        # Тождественная модель показывает тензор после встроенной нормализации.
        model = MobileModel(nn.Identity())
        actual = model(torch.from_numpy(image[None].astype(np.float32)))[0]
        expected = transform(False)(Image.fromarray(image))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_calibration_balanced_deterministic_and_grouped(self):
        rows = [dict(image_id=str(i), poi_id=str(i % 3), sha256=str(i), split_group=str(i // 3)) for i in range(60)]
        first = calibration_rows(rows, 12)
        self.assertEqual(first, calibration_rows(list(reversed(rows)), 12))
        self.assertEqual(len({r['split_group'] for r in first}), len(first))
        self.assertEqual({r['poi_id'] for r in first}, {'0', '1', '2'})
        self.assertTrue(all(r in rows for r in first))

    def test_threshold_ties_and_zero_allowance(self):
        for scores in ([.5] * 573, [.9, .8, .8, .1], [1., 1.]):
            threshold = calibrate(scores, .01)
            self.assertLessEqual(int((np.array(scores) >= threshold).sum()), int(len(scores) * .01))
        with self.assertRaises(ValueError):
            calibrate([], .01)

    def test_stable_softmax_and_quality_gate(self):
        np.testing.assert_allclose(softmax([[1000., 1000.]]), [[.5, .5]])
        self.assertTrue(quality_gate(dict(accuracy=.84, macro_f1=.82), dict(accuracy=.835, macro_f1=.815)))
        self.assertFalse(quality_gate(dict(accuracy=.84, macro_f1=.82), dict(accuracy=.84, macro_f1=.80)))

    def test_release_allows_only_public_artifacts_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'class_mapping.json').write_text(json.dumps({str(i): i for i in range(23)}))
            (root / 'poi_2gis_mapping.json').write_text(json.dumps({'entries': [dict(cv_poi_id=str(i), class_index=i) for i in range(23)]}))
            model = root / 'source.tflite'
            model.write_bytes(b'\0\0\0\0TFL3test-fixture')
            release = root / 'release'
            package(root, release, model, 'test-v1', 'fp32', {}, {'input': {}, 'output': {}}, {}, {})
            self.assertFalse(verify_release(release)['production_ready'])
            (release / 'model.tflite').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                verify_release(release)

    def test_reject_checkpoint_as_tflite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'checkpoint.pt'
            model.write_bytes(b'not a flatbuffer')
            with self.assertRaises(ValueError):
                package(root, root / 'release', model, 'v1', 'fp32', {}, {}, {}, {})

    def test_synthetic_controls_have_rgb_channel_order(self):
        samples = synthetic_inputs()
        self.assertEqual(samples.shape, (5, 224, 224, 3))
        np.testing.assert_array_equal(samples[0, 0, 0], [255, 0, 0])
        np.testing.assert_array_equal(samples[4, 20, 10], [10, 20, 255])


if __name__ == '__main__':
    unittest.main()
