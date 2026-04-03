import sys
import os
sys.path.insert(0, os.path.abspath('timesfm/src'))

import jax
import jax.numpy as jnp
from flax import nnx
from timesfm import TimesFM_2p5_200M_flax
from jax2onnx import to_onnx

print('Loading model...')
model = TimesFM_2p5_200M_flax.from_pretrained('google/timesfm-2.5-200m-flax')
print('Model loaded successfully.')

graphdef, state = nnx.split(model.model)

def stateless_fn(inputs, masks):
    m = nnx.merge(graphdef, state)
    return m(inputs, masks)

inputs = [
    jax.ShapeDtypeStruct(('B', 'N', 32), jnp.float32),
    jax.ShapeDtypeStruct(('B', 'N', 32), jnp.bool_)
]

print('Exporting to ONNX...')
to_onnx(
    stateless_fn,
    inputs=inputs,
    output_path='timesfm-2.5-200m-flax.onnx',
    return_mode='file',
    input_names=['inputs', 'masks'],
    output_names=['point_forecast', 'quantile_forecast']
)
print('Export complete')
