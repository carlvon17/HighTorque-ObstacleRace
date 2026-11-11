"""Export the audited Pi Plus Pro engine's FP32 tensors for CPU simulation.

This is deliberately restricted to one SHA-256, NOT a general TensorRT parser.
The layout was checked against the engine inspector and independent Jetson
inference. Never apply these byte offsets to another engine revision.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SHA256 = '49a64fc175dc9719554e246d42e2ccc0fa4dda97f672a66076a9e02dd8b97c13'
SHAPES = [(512, 345), (256, 512), (128, 256), (20, 128)]


def extract(raw):
    if hashlib.sha256(raw).hexdigest() != ENGINE_SHA256:
        raise ValueError('Unrecognized engine; obtain its original ONNX or audit its layout first')
    size = sum((m*n+m)*4 for m,n in SHAPES)
    if int.from_bytes(raw[217:225], 'little') != size:
        raise ValueError('Unexpected FP32 weight block')
    values = np.frombuffer(raw, dtype='<f4', count=size//4, offset=225)
    tensors, offset = {}, 0
    for layer, (rows, cols) in enumerate(SHAPES):
        weight = values[offset:offset+rows*cols]
        # TensorRT selected a transposed storage layout for the third GEMM.
        weight = weight.reshape(cols, rows).T if layer == 2 else weight.reshape(rows, cols)
        tensors[f'{layer*2}.weight'] = weight.copy()
        offset += rows*cols
        tensors[f'{layer*2}.bias'] = values[offset:offset+rows].copy()
        offset += rows
    return tensors


def predict(tensors, observations):
    x = np.asarray(observations, dtype=np.float32)
    for layer in range(4):
        x = x @ tensors[f'{layer*2}.weight'].T + tensors[f'{layer*2}.bias']
        if layer < 3:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=ROOT/'vendor/pi_plus_pro/amp.trt')
    parser.add_argument('--reference', type=Path, default=ROOT/'vendor/pi_plus_pro/engine_reference.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'vendor/pi_plus_pro/actor.npz')
    args = parser.parse_args()
    tensors = extract(args.engine.read_bytes())
    with np.load(args.reference, allow_pickle=False) as reference:
        actual = predict(tensors, reference['inputs'])
        expected = reference['outputs']
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        error = np.abs(actual-expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **tensors)
    report = {'engine_sha256': ENGINE_SHA256,
              'actor_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'samples': len(actual), 'max_absolute_error': float(error.max()),
              'mean_absolute_error': float(error.mean()), 'passed': True,
              'method': 'Exact FP32 tensors from pinned engine, checked against Jetson TensorRT outputs; no training or fitting.'}
    args.output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
