import argparse
import datetime
import os
import sys

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from torch import nn, Tensor
from pathlib import Path
import onnxruntime as ort
import uuid

pretest = os.environ.get("PRETEST", None) == "1"

input_shape = (8, 3, 224, 224)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@torch.no_grad()
def convert_ONNX(model: nn.Module, dummy_input, *, onnx_path: str):
    """
    Convert a PyTorch model to ONNX
    :param model: PyTorch model
    :param dummy_input: Dummy input tensor
    :param onnx_path: Path to save ONNX model
    :return:
    """
    previous_state = model.training

    model.eval()
    torch.onnx.export(model, dummy_input, onnx_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])
    print("")
    print("Model has been converted to ONNX")
    model.train(previous_state)


@torch.no_grad()
def _pytorch_throughput_latency(model: nn.Module, images: Tensor = None, *, mode: str,
                                type_: torch.dtype = torch.float32):
    """
    Measure the throughput and latency of a PyTorch model
    :param images: Input tensor
    :param model: PyTorch model
    :param mode: Mode of operation
    :param type_: Data type
    """
    assert mode.lower() in {"gpu", "cpu"}, mode

    previous_state = model.training
    previous_device = next(model.parameters()).device
    previous_type = next(model.parameters()).dtype

    device = torch.device("cuda") if mode.lower() == "gpu" else torch.device("cpu")

    if images is None:
        images = torch.randn(*input_shape, dtype=type_, device=device,
                             requires_grad=False)  # todo: verify if this matters.
    else:
        images = images.to(device, non_blocking=True)
        images = images.type(type_)

    model.eval()
    model.type(type_)
    model.to(device, non_blocking=True)

    batch_size = images.shape[0]

    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    # todo: check if it is necessary.
    if pretest:
        for _ in range(10):
            model(images)

    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    print(f"throughput averaged with: {batch_size} images in {mode} mode")
    tic1 = time.time()

    for _ in range(30):
        model(images)

    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    tic2 = time.time()
    exec_time = tic2 - tic1

    print(f"pytorch batch_size {batch_size} throughput: {(30 * batch_size) / exec_time:.2f}"
          f" with {str(device)}, {str(type_)}")
    print(f"pytorch batch_size {batch_size}    latency: {exec_time / (30 * batch_size):.4f}"
          f" with {str(device)}, {str(type_)}")

    model.train(previous_state)
    model.to(previous_device)
    model.type(previous_type)


def pytorch_throughput_latency(model: nn.Module, images: Tensor = None):
    _pytorch_throughput_latency(model=model, images=images, mode="gpu", type_=torch.float32)
    _pytorch_throughput_latency(model=model, images=images, mode="gpu", type_=torch.float16)

    _pytorch_throughput_latency(model=model, images=images, mode="cpu", type_=torch.float32)
    try:
        _pytorch_throughput_latency(model=model, images=images, mode="cpu", type_=torch.float16)
    except Exception as e:
        print(f"Failed to run on CPU with {e}", file=sys.stderr)


@torch.no_grad()
def _onnx_throughput_latency(model: nn.Module, images: Tensor = None, *, type_: torch.dtype = torch.float32):
    """
    Measure the throughput and latency of an ONNX model
    """

    previous_state = model.training
    previous_dtype = next(model.parameters()).dtype
    previous_device = next(model.parameters()).device

    device = torch.device("cuda") if ort.get_device() == "GPU" else torch.device("cpu")
    model.eval()
    model.type(type_)
    model.to(device, non_blocking=True)

    if images is None:
        images = torch.randn(*input_shape, dtype=type_, device=device, )
    else:
        images = images.to(device).type(type_)

    origin_result = model(images)

    os.makedirs("onnx", exist_ok=True)
    onnx_model_str = os.path.join("onnx", str(uuid.uuid4()))
    convert_ONNX(model, dummy_input=images, onnx_path=onnx_model_str)

    model.train(previous_state)
    model.type(previous_dtype)
    model.to(previous_device)

    ort_sess = ort.InferenceSession(onnx_model_str, providers=[
        "CPUExecutionProvider" if ort.get_device() == "CPU" else "CUDAExecutionProvider"])  #

    print(f"onnx device: {ort.get_device()}")

    x = to_numpy(images)
    batch_size = x.shape[0]

    if pretest:
        for _ in range(10):
            ort_sess.run(None, {'input': x})

    output = ort_sess.run(None, {"input": x})
    start_time = time.time()

    for _ in range(30):
        _ = ort_sess.run(None, {"input": x})
    exec_time = time.time() - start_time
    assert np.allclose(output[0], to_numpy(origin_result), rtol=1e-2, atol=1e-2)

    print(f"onnx batch_size {batch_size} throughput: {(30 * batch_size) / exec_time:.2f}"
          f" with {str(device)}, {str(type_)}")
    print(f"onnx batch_size {batch_size}    latency: {exec_time / (30 * batch_size):.2f}"
          f" with {str(device)}, {str(type_)}")


def onnx_throughput_latency(model: nn.Module, images: Tensor = None):
    _onnx_throughput_latency(model, images, type_=torch.float32)
    try:
        _onnx_throughput_latency(model, images, type_=torch.float16)
    except RuntimeError as e:
        print(f"Failed to run on {ort.get_device()}, torch.float16 with {e}", file=sys.stderr)
