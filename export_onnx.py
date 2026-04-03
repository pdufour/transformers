import os

import torch
import torch.nn as nn
from torch.export import Dim

from transformers.models.timesfm2_5.modeling_timesfm2_5 import TimesFm2_5ModelForPrediction


class TimesFmOnnxWrapper(nn.Module):
    """Single-tensor forward for ``torch.export`` / Dynamo ONNX (no list inputs)."""

    def __init__(self, model: TimesFm2_5ModelForPrediction, *, force_flip_invariance: bool):
        super().__init__()
        self.model = model
        self._force_flip_invariance = force_flip_invariance

    def forward(self, past_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ``TimesFm2_5ModelForPrediction`` API: one 1D series per batch item.
        out = self.model(
            [past_values],
            window_size=None,
            future_values=None,
            forecast_context_len=None,
            truncate_negative=False,
            force_flip_invariance=self._force_flip_invariance,
        )
        return out.mean_predictions, out.full_predictions


def export():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading model {model_id}...")

    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
    model.eval()

    # Flip-invariance runs the stack twice; disable for a smaller graph unless you need it.
    force_flip = bool(model.config.force_flip_invariance)
    wrapped = TimesFmOnnxWrapper(model, force_flip_invariance=force_flip)
    wrapped.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"ONNX wrapper force_flip_invariance={force_flip}")

    context_len = 1024
    past_values = torch.randn(context_len)

    os.makedirs("onnx", exist_ok=True)
    onnx_path = "onnx/model.onnx"

    # Dynamic 1D context length (dimension 0 of the single series tensor).
    seq_dim = Dim("sequence_length")
    dynamic_shapes = {"past_values": {0: seq_dim}}

    # Dynamo exporter targets opset >= 18; requesting 17 forced a brittle down-conversion. Use a
    # recent opset so the graph stays native (adjust if your runtime caps an older version).
    onnx_opset = 21

    print(f"Dynamo ONNX export (opset {onnx_opset}) -> {onnx_path} ...")

    torch.onnx.export(
        wrapped,
        (past_values,),
        onnx_path,
        input_names=["past_values"],
        output_names=["mean_predictions", "full_predictions"],
        opset_version=onnx_opset,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        external_data=False,
        do_constant_folding=True,
    )
    print("Export complete!")


if __name__ == "__main__":
    export()
