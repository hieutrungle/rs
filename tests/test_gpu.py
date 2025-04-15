#!/usr/bin/env python
"""Tests for `rs` package."""
# pylint: disable=redefined-outer-name

import pytest
import torch
import torchvision
import torchaudio
import sionna
import tensorflow as tf


@pytest.fixture(scope="module")
def check_tensorflow():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow is not available.")
    return tf


def test_tensorflow_gpu(check_tensorflow):
    """Test if TensorFlow GPU is available."""
    tf = check_tensorflow
    assert tf.test.is_gpu_available(), "TensorFlow GPU is not available."
    assert tf.test.is_built_with_cuda(), "TensorFlow is not built with CUDA support."
    # assert tf.test.is_built_with_rocm(), "TensorFlow is not built with ROCm support."


def test_torch_gpu():
    """Test if PyTorch GPU is available."""
    assert torch.cuda.is_available(), "PyTorch GPU is not available."
    assert torch.backends.cudnn.is_available(), "cuDNN is not available."
    # assert torch.backends.mps.is_available(), "Metal Performance Shaders (MPS) is not available."
    # assert torch.backends.mkldnn.is_available(), "MKL-DNN is not available."
    # assert torch.backends.openmp.is_available(), "OpenMP is not available."
    # assert torch.backends.mkl.is_available(), "MKL is not available."
