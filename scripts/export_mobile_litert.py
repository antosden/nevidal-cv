"""Linux: FP32 → статическая INT8 → оценка → пакет только после quality gates."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mobile_export import (SPLITS, load_model, verify_inputs, softmax, calibrate, synthetic_inputs,
                               metrics, quality_gate, package)
from src.small_classifier import write_json, digest


def tensor_info(detail):
    scale, zero = detail["quantization"]
    return {"name": detail["name"], "shape": detail["shape"].tolist(),
            "dtype": np.dtype(detail["dtype"]).name, "scale": float(scale), "zero_point": int(zero)}


def validate_io(detail):
    dtype = np.dtype(detail['dtype'])
    if dtype == np.dtype(np.float32):
        return
    if dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
        raise ValueError(f"Unsupported I/O dtype: {dtype}")
    scale, zero = detail['quantization']
    scales = detail.get('quantization_parameters', {}).get('scales', [])
    if len(scales) > 1:
        raise ValueError('Per-axis I/O quantization is not supported')
    if not np.isfinite(scale) or scale <= 0 or not np.iinfo(dtype).min <= zero <= np.iinfo(dtype).max:
        raise ValueError('Invalid I/O quantization scale/zero_point')


def encode_input(rgb, detail):
    """Квантуем исходные RGB 0..255, не выполняя ImageNet-нормализацию повторно."""
    validate_io(detail)
    if detail['dtype'] == np.float32:
        return np.asarray(rgb, dtype=np.float32)
    scale, zero = detail['quantization']
    values = np.asarray(rgb, dtype=np.float64) / scale + zero
    # Явное правило: ближайшее целое, половины от нуля (как Dart round).
    rounded = np.sign(values) * np.floor(np.abs(values) + .5)
    bounds = np.iinfo(detail['dtype'])
    return np.clip(rounded, bounds.min, bounds.max).astype(detail['dtype'])


def decode_output(values, detail):
    validate_io(detail)
    if detail['dtype'] == np.float32:
        return np.asarray(values, dtype=np.float32)
    scale, zero = detail['quantization']
    return ((np.asarray(values, dtype=np.float64) - zero) * scale).astype(np.float32)


def interpreter_for(path):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    interpreter = Interpreter(model_path=str(path), num_threads=4)
    backend, failure = "default_delegates", None
    try:
        interpreter.allocate_tensors()
    except RuntimeError as error:
        if "XNNPACK" not in str(error).upper():
            raise
        # Создаём новый interpreter: после неудачной делегации старый непригоден.
        failure = str(error)
        del interpreter
        print(f"XNNPACK failed for {path.name}; retrying builtin CPU kernels", flush=True)
        interpreter = Interpreter(model_path=str(path), num_threads=4,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors()
        backend = "builtin_without_default_delegates"
    interpreter._nevidal_runtime = {"backend": backend, "default_delegate_error": failure}
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Нужны один вход и один выход")
    if inputs[0]["shape"].tolist() != [1, 224, 224, 3] or outputs[0]["shape"].tolist() != [1, 23]:
        raise ValueError("Неверные размеры тензоров")
    for detail in (inputs[0], outputs[0]):
        validate_io(detail)
    return interpreter


def infer(path, images):
    interpreter = interpreter_for(path)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    logits = []
    for image in images:
        interpreter.set_tensor(input_detail["index"], encode_input(image[None], input_detail))
        interpreter.invoke()
        logits.append(decode_output(interpreter.get_tensor(output_detail["index"]), output_detail)[0].copy())
    contract = {"input": tensor_info(input_detail), "output": tensor_info(output_detail)}
    details = interpreter.get_tensor_details()
    contract["int8_tensors"] = sum(d["dtype"] == np.int8 for d in details)
    contract["float32_tensors"] = sum(d["dtype"] == np.float32 for d in details)
    contract["fully_integer_execution_verified"] = False
    contract["runtime"] = interpreter._nevidal_runtime
    print("Tensor contract: " + json.dumps(contract), flush=True)
    return np.stack(logits), contract


def quantize(fp32, output, calibration):
    from ai_edge_quantizer import quantizer, recipe, calibrator
    interpreter = interpreter_for(fp32)
    signatures = interpreter.get_signature_list()
    if len(signatures) != 1:
        raise ValueError("Ожидается одна signature для калибровки")
    signature, details = next(iter(signatures.items()))
    if len(details["inputs"]) != 1:
        raise ValueError("Неизвестная signature")
    input_name = details["inputs"][0]
    data = {signature: [{input_name: image[None].astype(np.float32)} for image in calibration]}
    qt = quantizer.Quantizer(str(fp32))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    result = qt.calibrate(data, mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED, num_threads=4)
    qt.quantize(calibration_result=result).export_model(str(output))


def run(work, output, version, reuse_models=None):
    if output.exists():
        raise FileExistsError(f"Нужен новый каталог результатов: {output}")
    output.mkdir(parents=True)
    status = {"status": "running", "completed_variants": [], "published": False}
    write_json(output / "status.json", status)
    try:
        verify_inputs(work)
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model, mapping = load_model(work)
        calibration = np.load(work / "calibration_rgb.npy", mmap_mode="r")
        sample = torch.from_numpy(calibration[:1].astype(np.float32))
        fp32 = output / "model_fp32.tflite"
        if reuse_models:
            import shutil
            from src.mobile_export import verify_release
            old_manifest = verify_release(reuse_models / "public" / version / "fp32")
            if digest(reuse_models / "model_fp32.tflite") != old_manifest["files"]["model.tflite"]["sha256"]:
                raise ValueError("Reused FP32 differs from validated package")
            for variant in ("fp32", "int8"):
                previous = reuse_models / f"model_{variant}.tflite"
                with previous.open("rb") as stream:
                    if stream.read(8)[4:8] != b"TFL3":
                        raise ValueError(f"Invalid reused FlatBuffer: {variant}")
                shutil.copyfile(previous, output / previous.name)
            print("Reusing existing FlatBuffers; evaluating again with updated runtime fallback", flush=True)
        else:
            import litert_torch
            print("Converting FP32", flush=True)
            litert_torch.convert(model, (sample,)).export(str(fp32))
        versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        write_json(output / "environment.json", versions)
        report = {"release_version": version, "selection_split": "val", "test_role": "diagnostic_only",
                  "max_validation_drop": .01, "variants": {},
                  "reused_model_sha256": {v: digest(output / f"model_{v}.tflite") for v in ("fp32", "int8")} if reuse_models else None,
                  "export_source_sha256": {name: digest(ROOT / name) for name in
                      ("src/mobile_export.py", "scripts/export_mobile_litert.py", "scripts/build_mobile_release_index.py")}}
        reference_val = metrics(work, "val", np.load(work / "val_reference_logits.npy"), 1., len(mapping))
        for variant in ("fp32", "int8"):
            path = output / f"model_{variant}.tflite"
            if variant == "int8" and not reuse_models:
                print("Calibrating INT8 on train only", flush=True)
                quantize(fp32, path, calibration)
            predictions, parity, contracts = {}, {}, {}
            # Сначала val: quality gates и threshold фиксируются до test.
            for split in ("val", "negative_val"):
                predictions[split], contracts[split] = infer(path, np.load(work / f"{split}_rgb.npy", mmap_mode="r"))
            threshold = calibrate(softmax(predictions["negative_val"]).max(1), .01)
            policy = {"version": f"{version}-{variant}-policy-v1", "score": "max_softmax",
                      "threshold": threshold, "comparison": ">=", "arithmetic": "float64 stable softmax",
                      "target_external_fpr": .01, "calibration_split": "negative_val",
                      "probability_calibrated": False,
                      "target_verification": "supported target AND argmax == target class_index AND score >= threshold; otherwise reject"}
            val = metrics(work, "val", predictions["val"], threshold, len(mapping))
            passed = quality_gate(reference_val, val)
            if variant == "int8":
                passed = passed and contracts["val"]["int8_tensors"] > 0 and path.stat().st_size < fp32.stat().st_size
            if variant == "fp32":
                for split in ("val", "negative_val"):
                    ref = np.load(work / f"{split}_reference_logits.npy")
                    error = float(np.max(np.abs(ref - predictions[split])))
                    agreement = float((ref.argmax(1) == predictions[split].argmax(1)).mean())
                    parity[split] = {"logit_max_abs_error": error, "top1_agreement": agreement}
                    passed = passed and error <= .001 and agreement == 1.
            # Записываем решение до просмотра диагностического test.
            write_json(output / f"{variant}_validation_gate.json", {"passed": passed, "val": val, "policy": policy, "parity": parity})
            if not passed:
                raise ValueError(f"{variant}: quality gate failed; см. {variant}_validation_gate.json")
            for split in ("test", "negative_test"):
                predictions[split], contracts[split] = infer(path, np.load(work / f"{split}_rgb.npy", mmap_mode="r"))
                if variant == "fp32":
                    ref = np.load(work / f"{split}_reference_logits.npy")
                    error = float(np.max(np.abs(ref - predictions[split])))
                    agreement = float((ref.argmax(1) == predictions[split].argmax(1)).mean())
                    parity[split] = {"logit_max_abs_error": error, "top1_agreement": agreement}
                    if error > .001 or agreement != 1.:
                        raise ValueError(f"FP32 test parity failed: {split}")
            evaluation = {s: metrics(work, s, predictions[s], threshold, len(mapping)) for s in SPLITS}
            evaluation["parity"] = parity
            evaluation["tensor_contract"] = contracts["val"]
            evaluation["bytes"] = path.stat().st_size
            evaluation["validation_gate_passed"] = True
            report["variants"][variant] = evaluation
            write_json(output / "comparison.json", report)
            for split in SPLITS:
                np.save(output / f"{variant}_{split}_logits.npy", predictions[split])
            synthetic_logits, _ = infer(path, synthetic_inputs())
            conformance = {
                "purpose": "synthetic tensor parity only; no evidence of photo quality",
                "shape": [1, 224, 224, 3], "logical_rgb_dtype": "uint8",
                "tensor_input": contracts["val"]["input"], "tensor_output": contracts["val"]["output"],
                "expected_logits_semantics": "dequantized float32 logits", "logit_atol": .002,
                "patterns": ["solid RGB(255,0,0)", "solid RGB(0,255,0)", "solid RGB(0,0,255)",
                             "solid RGB(124,116,104)", "R=x; G=y; B=255*((x//16+y//16)%2); x,y=0..223"],
                "expected_logits": synthetic_logits.tolist(),
                "expected_class_indices": synthetic_logits.argmax(1).tolist()}
            package(work, output / "public" / version / variant, path, version, variant, policy, contracts["val"], evaluation, conformance)
            status["completed_variants"].append(variant)
            write_json(output / "status.json", status)
        status["status"] = "validated_integration_candidates"
    except Exception as error:
        status.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "status.json", status)
    try:
        from scripts.build_mobile_release_index import build_index
        build_index(output)
    except Exception as error:
        status.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="poi-small-2026-10-30-v1")
    parser.add_argument("--reuse-models", type=Path, help="Previous failed result with validated FP32 and exported INT8")
    args = parser.parse_args()
    run(args.work.resolve(), args.output.resolve(), args.version, args.reuse_models.resolve() if args.reuse_models else None)
