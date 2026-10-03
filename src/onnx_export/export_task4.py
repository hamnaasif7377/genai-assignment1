"""
Export the trained Task 4 generator to ONNX and verify against PyTorch.
Only the generator is needed for inference; the discriminator is a
training-only component and is not exported.
"""
import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.task4_gan import UNetGenerator


def run_export(args):
    with open(args.best_params, encoding="utf-8-sig") as f:
        best_params = json.load(f)

    model = UNetGenerator(
        base_ch=best_params["base_ch"],
        style_dim=best_params["style_dim"],
        dropout=False,  # inference mode, dropout disabled regardless of training config
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_photo = torch.randn(1, 3, 128, 128)
    dummy_style = torch.tensor([0], dtype=torch.int64)

    torch.onnx.export(
        model, (dummy_photo, dummy_style), args.onnx_out,
        input_names=["photo", "style"], output_names=["sketch"],
        dynamic_axes={"photo": {0: "batch_size"}, "style": {0: "batch_size"}, "sketch": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {args.onnx_out}")

    onnx_model = onnx.load(args.onnx_out)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid")

    with torch.no_grad():
        torch_output = model(dummy_photo, dummy_style).numpy()

    ort_session = ort.InferenceSession(args.onnx_out)
    onnx_output = ort_session.run(None, {"photo": dummy_photo.numpy(), "style": dummy_style.numpy()})[0]

    max_diff = np.max(np.abs(torch_output - onnx_output))
    print(f"Max absolute difference (PyTorch vs ONNX): {max_diff:.8f}")
    print("PASS" if max_diff < 1e-4 else "WARNING: exceeds tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/GenAI_A1_Task4/checkpoints/task4_generator_best.pt")
    parser.add_argument("--best_params", default="configs/task4_best_params.json")
    parser.add_argument("--onnx_out", default="/content/drive/MyDrive/GenAI_A1_Task4/onnx/task4_generator.onnx")
    args = parser.parse_args()
    run_export(args)
