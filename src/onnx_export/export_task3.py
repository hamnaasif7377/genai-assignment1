"""Export the trained Task 3 soft MoE pipeline (gate + 3 experts, as one
combined model) to ONNX, verified against PyTorch."""
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.task3_soft_moe import SoftMoERestoration


def run_export(args):
    with open(args.classifier_best_params, encoding="utf-8-sig") as f:
        clf_params = json.load(f)
    with open(args.specialists_best_params, encoding="utf-8-sig") as f:
        spec_params = json.load(f)
    with open(args.task3_best_params, encoding="utf-8-sig") as f:
        moe_params = json.load(f)

    model = SoftMoERestoration(
        clf_base_ch=clf_params["base_ch"], clf_dropout=clf_params["dropout"],
        spec_base_ch=spec_params["base_ch"], spec_bottleneck_dim=spec_params["bottleneck_dim"],
        spec_dropout=spec_params["dropout"], temperature=moe_params["temperature"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)

    os.makedirs(os.path.dirname(args.onnx_out), exist_ok=True)
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
    print("PASS" if max_diff < 1e-4 else "WARNING: exceeds tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/GenAI_A1/checkpoints/task3_soft_moe_best.pt")
    parser.add_argument("--classifier_best_params", default="configs/task2_classifier_best_params.json")
    parser.add_argument("--specialists_best_params", default="configs/task2_specialists_best_params.json")
    parser.add_argument("--task3_best_params", default="configs/task3_best_params.json")
    parser.add_argument("--onnx_out", default="/content/drive/MyDrive/GenAI_A1/onnx/task3_soft_moe.onnx")
    args = parser.parse_args()
    run_export(args)
