"""
Export all Task 2 models (classifier + 3 specialists) to ONNX,
verifying each against its PyTorch output.
"""
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.task2_classifier import CorruptionClassifier
from src.models.task2_specialists import SpecialistAutoencoder


def export_and_verify(model, dummy_input, onnx_path, input_names, output_names):
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}},
        opset_version=17,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {input_names[0]: dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_output - onnx_output))
    status = "PASS" if max_diff < 1e-4 else "WARNING"
    print(f"  {os.path.basename(onnx_path)}: max_diff={max_diff:.8f} [{status}]")
    return max_diff


def run_export(args):
    os.makedirs(args.onnx_dir, exist_ok=True)
    dummy_input = torch.randn(1, 3, 128, 128)
    results = {}

    print("Exporting classifier...")
    with open(args.classifier_best_params, encoding="utf-8-sig") as f:
        clf_params = json.load(f)
    classifier = CorruptionClassifier(base_ch=clf_params["base_ch"], dropout=clf_params["dropout"])
    classifier.load_state_dict(torch.load(f"{args.checkpoint_root}/task2_classifier_best.pt", map_location="cpu"))
    classifier.eval()
    results["classifier"] = export_and_verify(
        classifier, dummy_input, f"{args.onnx_dir}/task2_classifier.onnx",
        ["input"], ["logits"]
    )

    with open(args.specialists_best_params, encoding="utf-8-sig") as f:
        spec_params = json.load(f)

    for ctype in ["salt_pepper", "blur", "occlusion"]:
        print(f"Exporting {ctype} specialist...")
        model = SpecialistAutoencoder(
            base_ch=spec_params["base_ch"],
            bottleneck_dim=spec_params["bottleneck_dim"],
            dropout=spec_params["dropout"],
        )
        model.load_state_dict(torch.load(
            f"{args.checkpoint_root}/task2_specialist_{ctype}_best.pt", map_location="cpu"
        ))
        model.eval()
        results[ctype] = export_and_verify(
            model, dummy_input, f"{args.onnx_dir}/task2_specialist_{ctype}.onnx",
            ["input"], ["output"]
        )

    print("\nAll Task 2 models exported successfully.")
    print("Max differences:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1/checkpoints")
    parser.add_argument("--classifier_best_params", default="configs/task2_classifier_best_params.json")
    parser.add_argument("--specialists_best_params", default="configs/task2_specialists_best_params.json")
    parser.add_argument("--onnx_dir", default="/content/drive/MyDrive/GenAI_A1/onnx")
    args = parser.parse_args()
    run_export(args)
