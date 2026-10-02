"""
Export the trained Task 1 model to ONNX and verify PyTorch/ONNX outputs match.
"""
import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.task1_autoencoder import UniversalAutoencoder


def run_export(args):
    with open(args.best_params, encoding="utf-8-sig") as f:
        best_params = json.load(f)

    model = UniversalAutoencoder(
        base_ch=best_params["base_ch"],
        bottleneck_dim=best_params["bottleneck_dim"],
        dropout=best_params["dropout"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)

    torch.onnx.export(
        model, dummy_input, args.onnx_out,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {args.onnx_out}")

    onnx_model = onnx.load(args.onnx_out)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid")

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(args.onnx_out)
    onnx_output = ort_session.run(None, {"input": dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_output - onnx_output))
    print(f"Max absolute difference (PyTorch vs ONNX): {max_diff:.8f}")

    if max_diff < 1e-4:
        print("PASS: ONNX output matches PyTorch output within tolerance")
    else:
        print("WARNING: difference exceeds tolerance, investigate before using ONNX model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/GenAI_A1/checkpoints/task1_final_best.pt")
    parser.add_argument("--best_params", default="configs/task1_best_params.json")
    parser.add_argument("--onnx_out", default="/content/drive/MyDrive/GenAI_A1/onnx/task1_universal_autoencoder.onnx")
    args = parser.parse_args()
    run_export(args)
