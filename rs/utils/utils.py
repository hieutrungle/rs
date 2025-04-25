import yaml
import re
import os
import shutil
import math
from typing import Tuple
import GPUtil
import numpy as np


def mkdir_not_exists(folder_dir: str) -> None:
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def mkdir_with_replacement(folder_dir: str) -> None:
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)
    os.makedirs(folder_dir)


def create_filename(dir: str, filename: str) -> str:
    """Create a filename in the given directory based on the last numbered image file."""
    mkdir_not_exists(dir)
    base_name, ext = os.path.splitext(filename)
    max_index = 0

    # Find the last numbered image file
    for file in os.listdir(dir):
        if file.endswith(ext):
            match = re.search(r"_(\d+)$", os.path.splitext(file)[0])
            if match:
                max_index = max(max_index, int(match.group(1)))

    # Increment the index and create the new filename
    new_index = max_index + 1
    new_filename = f"{base_name}_{new_index:05d}{ext}"
    return os.path.join(dir, new_filename)


def load_yaml_file(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def write_yaml_file(file_path: str, data: dict) -> None:
    tmp_file = file_path.split(".")[0] + "_tmp.yaml"
    with open(tmp_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    os.rename(tmp_file, file_path)


def load_config(config_file: str) -> dict:
    config_kwargs = load_yaml_file(config_file)
    for k, v in config_kwargs.items():
        if isinstance(v, str):
            if v.lower() == "true":
                config_kwargs[k] = True
            elif v.lower() == "false":
                config_kwargs[k] = False
            elif v.isnumeric():
                config_kwargs[k] = float(v)
            elif re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", v):
                config_kwargs[k] = float(v)
            else:
                config_kwargs[k] = v

    config = config_kwargs
    return config


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def spherical2cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def track_gpu_usage():
    gpus = GPUtil.getGPUs()
    gpus_usage = [gpu.load for gpu in gpus]
    gpus_memory = [gpu.memoryUtil for gpu in gpus]
    return np.mean(gpus_usage) * 100, np.mean(gpus_memory) * 100


# Logging
def log_args(args) -> None:
    """Logs arguments to the console."""
    print(f"{'*'*23} ARGS BEGIN {'*'*23}")
    message = ""
    for k, v in args.__dict__.items():
        if isinstance(v, str):
            message += f"{k} = '{v}'\n"
        else:
            message += f"{k} = {v}\n"
    print(f"{message}")
    print(f"{'*'*24} ARGS END {'*'*24}\n")


def log_config(config: dict) -> None:
    """Logs configuration to the console."""
    print(f"{'*'*23} CONFIG BEGIN {'*'*23}")
    message = ""
    for k, v in config.items():
        if isinstance(v, str):
            message += f"{k} = '{v}'\n"
        else:
            message += f"{k} = {v}\n"
    print(f"{message}")
    print(f"{'*'*24} CONFIG END {'*'*24}\n")
