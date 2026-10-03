"""
Loads and caches ONNX Runtime inference sessions for each task's model.
Models are loaded once at application startup and reused across requests,
since loading an ONNX session has real overhead and should not happen
on every single API call.
"""
import os

import onnxruntime as ort

ONNX_DIR = os.path.join(os.path.dirname(__file__), "onnx_models")

_sessions = {}


def get_session(model_filename: str) -> ort.InferenceSession:
    if model_filename not in _sessions:
        model_path = os.path.join(ONNX_DIR, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Make sure it has been copied into backend/app/onnx_models/."
            )
        _sessions[model_filename] = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    return _sessions[model_filename]
