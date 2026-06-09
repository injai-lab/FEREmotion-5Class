from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


NUM_CLASSES = 5
IMG_SIZE = 160
REPO_ROOT = Path(__file__).resolve().parents[2]


class FaceNetFER5Class(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631)
        backbone.classify = False
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(x), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FaceNet 5-class FER model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "models" / "facenet_5class_best.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "android_project" / "app" / "src" / "main" / "assets" / "facenet-5class.onnx",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Pass --checkpoint or place facenet_5class_best.pth in the repo models folder."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    model = FaceNetFER5Class()
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    wrapped = ProbabilityModel(model).eval()

    sample = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    print(f"Exporting FaceNet 5-class checkpoint: {args.checkpoint}")
    print(f"Target: {args.output}")
    torch.onnx.export(
        wrapped,
        sample,
        args.output,
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    output = session.run(["probabilities"], {"input": sample.numpy()})[0]
    if output.shape != (1, NUM_CLASSES):
        raise RuntimeError(f"Unexpected ONNX output shape: {output.shape}")

    print("FACENET_5CLASS_ONNX_EXPORT_OK")
    print(f"bytes={args.output.stat().st_size}")
    print(f"output_shape={output.shape}")
    print(f"probability_sum={float(output.sum()):.6f}")


if __name__ == "__main__":
    main()
