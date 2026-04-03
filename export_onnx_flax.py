import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("timesfm/src"))

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# jax2onnx: patched LinearGeneral binds (x, kernel, bias) even when bias is
# None (TimesFM uses use_bias=False). JAX cannot bind None; use dot_general.
# ---------------------------------------------------------------------------
from jax2onnx.plugins.flax.nnx.linear_general import LinearGeneralPlugin


def _linear_general_make_patch_fixed(orig_fn):
    LinearGeneralPlugin._ORIGINAL_CALL = orig_fn
    prim = LinearGeneralPlugin._PRIM

    def patched(self, x):
        rank = max(getattr(x, "ndim", len(x.shape)), 1)
        if isinstance(self.axis, int):
            lhs = (self.axis % rank,)
        else:
            lhs = tuple((a % rank) for a in self.axis)
        rhs = tuple(range(len(self.in_features)))
        dn = ((lhs, rhs), ((), ()))
        kernel = jnp.asarray(self.kernel)
        bias = jnp.asarray(self.bias) if self.bias is not None else None
        if bias is None:
            return jax.lax.dot_general(x, kernel, dimension_numbers=dn)
        return prim.bind(x, kernel, bias, dimension_numbers=dn)

    return patched


LinearGeneralPlugin._make_patch = _linear_general_make_patch_fixed

# ---------------------------------------------------------------------------
# TimesFM Flax: nnx.scan on stacked transformers clashes with nested JAX trace
# (jax.make_jaxpr / jax2onnx). Replace with explicit layer loop for export only.
# ---------------------------------------------------------------------------
import timesfm.timesfm_2p5.timesfm_2p5_flax as _tf25_flax


def _slice_stacked_transformer_state(state, layer_idx: int):
    def maybe_slice(x):
        if isinstance(x, nnx.Param):
            return nnx.Param(jnp.asarray(x[layer_idx]))
        return x

    return jax.tree_util.tree_map(
        maybe_slice, state, is_leaf=lambda x: isinstance(x, nnx.Param)
    )


def _apply_stacked_transformers_unroll(stacked_xf, x, m, decode_cache=None):
    graph, state = nnx.split(stacked_xf)
    if decode_cache is None:
        leaves = jax.tree.leaves(state, is_leaf=lambda z: isinstance(z, nnx.Param))
        num_layers = int(leaves[0].shape[0])
        decode_cache = [None] * num_layers
    else:
        num_layers = len(decode_cache)
    new_decode_cache = []
    for i in range(num_layers):
        layer_i = nnx.merge(graph, _slice_stacked_transformer_state(state, i))
        x, dc = layer_i(x, m, decode_cache=decode_cache[i])
        new_decode_cache.append(dc)
    return x, new_decode_cache


_tf25_flax._apply_stacked_transformers = _apply_stacked_transformers_unroll

# jax2onnx's softplus primitive does not accept nnx.Param; pass a JAX array.
from timesfm.flax import transformer as _tf_transformer


def _per_dim_scale_call_export(self, x):
    w = jnp.asarray(self.per_dim_scale)
    return x * (1.442695041 / jnp.sqrt(self.num_dims) * jax.nn.softplus(w))


_tf_transformer.PerDimScale.__call__ = _per_dim_scale_call_export

from jax2onnx import to_onnx
from timesfm import TimesFM_2p5_200M_flax

# Trace with concrete patch counts: symbolic N breaks `jnp.arange(n_patches)` inside RoPE.
# Re-export with different (batch, num_patches) if you need another fixed size.
TRACE_BATCH = 1
TRACE_NUM_PATCHES = 16

print("Loading model...")
model = TimesFM_2p5_200M_flax.from_pretrained("google/timesfm-2.5-200m-flax")
print("Model loaded successfully.")


def _forward_for_export(inputs, masks):
    (_in_emb, _out_emb, point_forecast, quantile_forecast), _decode_cache = model.model(
        inputs, masks
    )
    return point_forecast, quantile_forecast

inputs = [
    jax.ShapeDtypeStruct((TRACE_BATCH, TRACE_NUM_PATCHES, 32), jnp.float32),
    jax.ShapeDtypeStruct((TRACE_BATCH, TRACE_NUM_PATCHES, 32), jnp.bool_),
]

print("Exporting to ONNX...")
_onnx_path = Path("timesfm-2.5-200m-flax.onnx")
_onnx_path.unlink(missing_ok=True)
Path("timesfm-2.5-200m-flax.onnx.data").unlink(missing_ok=True)
to_onnx(
    _forward_for_export,
    inputs=inputs,
    output_path=str(_onnx_path),
    return_mode="file",
    input_names=["inputs", "masks"],
    output_names=["point_forecast", "quantile_forecast"],
)
print("Export complete: timesfm-2.5-200m-flax.onnx")
