#!/usr/bin/env python
"""Tests for `rs` package."""
# pylint: disable=redefined-outer-name

import pytest


def test_tensorflow_import():
    """Test if TensorFlow is imported correctly."""
    try:
        import tensorflow as tf
    except ImportError as e:
        pytest.fail(f"TensorFlow import failed: {e}")


def test_torch_import():
    """Test if PyTorch is imported correctly."""
    try:
        import torch
    except ImportError as e:
        pytest.fail(f"PyTorch import failed: {e}")


def test_torchvision_import():
    """Test if torchvision is imported correctly."""
    try:
        import torchvision
    except ImportError as e:
        pytest.fail(f"torchvision import failed: {e}")


def test_torchaudio_import():
    """Test if torchaudio is imported correctly."""
    try:
        import torchaudio
    except ImportError as e:
        pytest.fail(f"torchaudio import failed: {e}")


def test_sionna_import():
    """Test if Sionna is imported correctly."""
    try:
        import sionna
    except ImportError as e:
        pytest.fail(f"Sionna import failed: {e}")
