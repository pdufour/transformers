import os

import torch
import torch.nn as nn
from torch.export import Dim

from transformers.models.timesfm2_5.modeling_timesfm2_5 import TimesFm2_5ModelForPrediction


class TimesFmOnnxWrapper(nn.Module):
    """ONNX sees `[batch, time]`; each row is turned into a 1D series for ``forward``."""

    def __init__(self, model: TimesFm2_5ModelForPrediction) -> None:
        super().__init__()
        self.model = model

    def forward(self, past_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        series = list(past_values.unbind(0))
        out = self.model(series)
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

    # Use batch size 2 for the example to see if it generalizes
    batch_size = 2
    context_len = 512
    past_values = torch.randn(batch_size, context_len)

    os.makedirs("onnx", exist_ok=True)
    onnx_path = "onnx/model.onnx"

    # Define dynamic shapes for Dynamo
    batch = Dim("batch", min=1, max=1024)
    sequence = Dim("sequence", min=1, max=16384)
    dynamic_shapes = {"past_values": {0: batch, 1: sequence}}

    onnx_opset = 21  # Latest ONNX opset for maximum feature support

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
        external_data=True,
        do_constant_folding=True,
        optimize=False,
    )
    print("Export complete!")


if __name__ == "__main__":
    export()
