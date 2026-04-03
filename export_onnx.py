import os

import torch
import torch.nn as nn
from torch.export import Dim

from transformers.models.timesfm2_5.modeling_timesfm2_5 import TimesFm2_5ModelForPrediction


class TimesFmOnnxWrapper(nn.Module):
    """Batched `[batch, time]` input for ONNX; kwargs defer to ``TimesFm2_5ModelForPrediction.forward`` defaults."""

    def __init__(self, model: TimesFm2_5ModelForPrediction) -> None:
        super().__init__()
        self.model = model

    def forward(self, past_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(past_values)
        return out.mean_predictions, out.full_predictions


def export():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading model {model_id}...")

    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
    model.eval()

    wrapped = TimesFmOnnxWrapper(model)
    wrapped.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    batch_size = 2
    context_len = 1024
    past_values = torch.randn(batch_size, context_len)

    os.makedirs("onnx", exist_ok=True)
    onnx_path = "onnx/model.onnx"

    batch_dim = Dim("batch_size")
    seq_dim = Dim("sequence_length")
    dynamic_shapes = {"past_values": {0: batch_dim, 1: seq_dim}}

    onnx_opset = 21

    fast_export = os.environ.get("ONNX_EXPORT_OPTIMIZE", "").lower() not in ("1", "true", "yes")
    print(
        f"Dynamo ONNX export (opset {onnx_opset}, optimize={'ON' if not fast_export else 'OFF'}) -> {onnx_path} ..."
    )

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
        optimize=not fast_export,
    )
    print("Export complete!")


if __name__ == "__main__":
    export()
