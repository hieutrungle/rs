import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation

from typing import Tuple, Optional
from collections import OrderedDict
import subprocess
import time
import numpy as np
import pickle
import glob
import math
import copy
from rs.utils import utils
from rs.blender_script import shared_utils
import tensorflow as tf
from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, TensorDict
import torch


class SharedAPV2(EnvBase):

    def __init__(
        self,
        idx: int,
        sionna_config_file: str,
        log_string: str = "SharedAPV2",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.idx = idx
        self.log_string = log_string
        self.seed = seed + idx
        self.np_rng = np.random.default_rng(self.seed)

        tf.random.set_seed(self.seed)
        print(f"using GPU: {tf.config.experimental.list_physical_devices('GPU')}")

        self.sionna_config = utils.load_config(sionna_config_file)

        # sigmap
        self.sig_cmap = sigmap.engine.SignalCoverageMap(self.sionna_config)

        # positions
        self.rx_pos = np.array(self.sionna_config["rx_positions"], dtype=np.float32)
        self.rt_pos = np.array(self.sionna_config["rt_positions"], dtype=np.float32)
        self.tx_pos = np.array(self.sionna_config["tx_positions"], dtype=np.float32)

        # orient the tx
        tx_orientations = []
        for i in range(len(self.rt_pos)):
            r, theta, phi = compute_rot_angle(self.tx_pos[i], self.rt_pos[i])
            tx_orientations.append([phi, theta - math.pi / 2, 0.0])
        self.sionna_config["tx_orientations"] = tx_orientations

        # Set up logging
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Load the reflector configuration, angle is in radians
        reflector_configs = shared_utils.get_config_shared_ap()
        self.theta_configs = reflector_configs[0]
        self.phi_configs = reflector_configs[1]
        self.num_groups = reflector_configs[2]
        self.num_elements_per_group = reflector_configs[3]

        # angles = [theta, phi] for each tile
        # theta: zenith angle, phi: azimuth angle
        self.init_thetas = []
        self.init_phis = []
        self.angle_spaces = []
        theta_highs = []
        phi_highs = []
        theta_lows = []
        phi_lows = []

        for i in range(len(self.rt_pos)):
            init_theta = self.theta_configs[0][i]
            init_phi = self.phi_configs[0][i]

            theta_high = self.theta_configs[2][i]
            phi_high = self.phi_configs[2][i]
            per_group_high = [phi_high] + [theta_high] * self.num_elements_per_group
            angle_high = np.concatenate([per_group_high] * self.num_groups)
            theta_low = self.theta_configs[1][i]
            phi_low = self.phi_configs[1][i]
            per_group_low = [phi_low] + [theta_low] * self.num_elements_per_group
            angle_low = np.concatenate([per_group_low] * self.num_groups)
            self.angle_spaces.append((angle_low, angle_high))

            # storage
            self.init_thetas.append(init_theta)
            self.init_phis.append(init_phi)
            theta_highs.append(theta_high)
            phi_highs.append(phi_high)
            theta_lows.append(theta_low)
            phi_lows.append(phi_low)

        global_angle_low = np.concatenate([space[0] for space in self.angle_spaces])
        global_angle_high = np.concatenate([space[1] for space in self.angle_spaces])

        # position space
        self.position_spaces = []
        for i in range(len(self.rt_pos)):
            low = -100.0
            high = 100.0
            shape = (len(np.concatenate([self.rx_pos, self.rt_pos[i : i + 1]]).flatten()),)
            self.position_spaces.append((low, high, shape))

        global_position_low = -100.0
        global_position_high = 100.0
        global_position_shape = (len(np.concatenate([self.rx_pos, self.rt_pos]).flatten()),)

        # focal vecs space <-> action space
        # Action is a changes in focals [delta_r, delta_theta, _delta_phi] for each group
        # focals = [r, theta, phi] for each group
        self._initialize_focal_spaces(theta_highs, phi_highs, theta_lows, phi_lows)

        # Observation spec
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                low=torch.tensor(
                    np.concatenate(
                        [global_angle_low, np.full(global_position_shape, global_position_low)]
                    ),
                    dtype=torch.float32,
                ),
                high=torch.tensor(
                    np.concatenate(
                        [global_angle_high, np.full(global_position_shape, global_position_high)]
                    ),
                    dtype=torch.float32,
                ),
                shape=(len(global_angle_low) + len(global_position_shape),),
                dtype=torch.float32,
            )
        )

        # Action spec
        action_space_shape = tuple((3 * self.num_groups,))
        action_low = np.concatenate(
            [-1.0 * np.ones(action_space_shape) for _ in range(len(self.rt_pos))]
        )
        action_high = np.concatenate(
            [1.0 * np.ones(action_space_shape) for _ in range(len(self.rt_pos))]
        )
        self.action_spec = BoundedTensorSpec(
            low=torch.tensor(action_low, dtype=torch.float32),
            high=torch.tensor(action_high, dtype=torch.float32),
            shape=(len(action_low),),
            dtype=torch.float32,
        )

        # Reward set up
        self.taken_steps = 0.0
        self.prev_gains = [0.0 for _ in range(len(self.rx_pos))]
        self.cur_gains = [0.0 for _ in range(len(self.rx_pos))]

        # range for new rx positions
        self.obstacle_pos = [[2.37, -17.656], [-2.6, -11.945], [1.86, -10.7654], [-1.71, -15.6846]]

        # range for new rx positions: [[x_min, x_max], [y_min, y_max]]
        self.rx_pos_ranges = [
            [[0.2, 3.5], [-1.0, 2.0]],
            [[0.2, 3.5], [-5.5, -1.5]],
            [[-3.5, -0.2], [-1.0, 2.0]],
            [[-3.5, -0.2], [-5.5, -1.5]],
        ]

        # range for rt positions: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.rt_pos_delta_ranges = [
            [[-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2]],
            [[-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2]],
        ]

        self.default_sionna_config = copy.deepcopy(self.sionna_config)

        self.eval_mode = False

    def _initialize_focal_spaces(self, theta_highs, phi_highs, theta_lows, phi_lows):
        r_high = 40.0
        r_low = 5.0
        self.focal_spaces = []
        for i in range(len(self.rt_pos)):
            theta_high = theta_highs[i]
            theta_low = theta_lows[i]
            phi_high = phi_highs[i]
            phi_low = phi_lows[i]
            self.focal_spaces.append(
                (
                    np.asarray([r_low, theta_low, phi_low] * self.num_groups),
                    np.asarray([r_high, theta_high, phi_high] * self.num_groups),
                )
            )

    def reset(self, tensordict=None, **kwargs):
        self.sionna_config = copy.deepcopy(self.default_sionna_config)

        # reset rx_pos
        rx_pos = self._prepare_rx_positions()
        self.sionna_config["rx_positions"] = rx_pos
        self.rx_pos = np.array(rx_pos, dtype=np.float32)

        # reset rt_pos
        rt_pos_deltas = self._prepare_rt_delta_positions()
        self.rt_pos = np.array(self.rt_pos, dtype=np.float32) + rt_pos_deltas
        self.sionna_config["rt_positions"] = list(self.rt_pos)

        # orient the tx
        tx_orientations = []
        for i in range(len(self.rt_pos)):
            r, theta, phi = compute_rot_angle(self.tx_pos[i], self.rt_pos[i])
            tx_orientations.append([phi, theta - math.pi / 2, 0.0])
        self.sionna_config["tx_orientations"] = tx_orientations

        start_init = False
        if kwargs is not None:
            start_init = kwargs.get("start_init", False)
            print(f"\nRESET with start_init: {start_init}")

        self.focals = [None for _ in range(len(self.rt_pos))]
        for i in range(len(self.rt_pos)):
            low = self.focal_spaces[i][0]
            high = self.focal_spaces[i][1]
            center = (low + high) / 2.0
            low = center - (center - low) * 0.9
            high = center + (high - center) * 0.9
            if start_init:
                self.focals[i] = self.np_rng.uniform(low=low, high=high)
            else:
                self.focals[i] = self.np_rng.normal(
                    loc=(low + high) / 2.0, scale=abs(high - low) / 6.0
                )
            self.focals[i] = np.clip(self.focals[i], low, high)
        self.focals = np.asarray(self.focals, dtype=np.float32)

        self.angles = self._blender_step(self.focals)
        for i in range(len(self.angle_spaces)):
            self.angles[i] = np.asarray(self.angles[i], dtype=np.float32)
            self.angles[i] = np.clip(
                self.angles[i], self.angle_spaces[i][0], self.angle_spaces[i][1]
            )

        if kwargs and "eval_mode" in kwargs:
            self.eval_mode = kwargs["eval_mode"]
        self.prev_gains = self._run_sionna_dB(self.eval_mode)
        self.cur_gains = self.prev_gains

        global_angles = self.angles.flatten()
        global_positions = np.concatenate([self.rx_pos, self.rt_pos], axis=0).flatten()
        observation = np.concatenate([global_angles, global_positions], axis=-1).flatten()

        reward = torch.zeros(1, dtype=torch.float32)
        done = torch.zeros(1, dtype=torch.bool)

        next_tensordict = TensorDict(
            {
                "observation": torch.tensor(observation, dtype=torch.float32),
                "reward": reward,
                "done": done,
            },
            batch_size=[1],
        )

        self.taken_steps = 0.0

        return next_tensordict

    def step(self, tensordict):
        self.taken_steps += 1.0
        self.prev_gains = self.cur_gains

        action = tensordict.get("action").numpy()
        action = np.reshape(action, (len(self.rt_pos), -1))
        tmp = np.reshape(action, (len(self.rt_pos) * self.num_groups, 3))
        tmp[:, 0] = tmp[:, 0]
        tmp[:, 1] = np.deg2rad(tmp[:, 1])
        tmp[:, 2] = np.deg2rad(tmp[:, 2])
        action = np.reshape(tmp, self.focals.shape)

        self.focals = self.focals + action

        low = np.asarray([space[0] for space in self.focal_spaces])
        high = np.asarray([space[1] for space in self.focal_spaces])
        local_out_of_bounds = [
            np.sum((focal < low[i]) + (focal > high[i]), dtype=np.float32)
            for i, focal in enumerate(self.focals)
        ]
        out_of_bounds = np.sum(local_out_of_bounds, dtype=np.float32)

        self.focals = np.clip(self.focals, low, high)

        self.angles = self._blender_step(self.focals)
        self.angles = np.asarray(self.angles, dtype=np.float32)

        low = np.asarray([space[0] for space in self.angle_spaces])
        high = np.asarray([space[1] for space in self.angle_spaces])
        if np.any(self.angles < low) or np.any(self.angles > high):
            print("Warning: angles out of bounds")

        truncated = False
        if self.taken_steps > 100:
            truncated = True
        terminated = False
        self.cur_gains = self._run_sionna_dB(eval_mode=self.eval_mode)

        global_reward = self._cal_reward(
            np.mean(self.prev_gains, axis=1), np.mean(self.cur_gains, axis=1), out_of_bounds
        )

        global_angles = self.angles.flatten()
        global_positions = np.concatenate([self.rx_pos, self.rt_pos], axis=0).flatten()
        next_observation = np.concatenate([global_angles, global_positions], axis=-1).flatten()

        next_tensordict = TensorDict(
            {
                "observation": torch.tensor(next_observation, dtype=torch.float32),
                "reward": torch.tensor(global_reward, dtype=torch.float32),
                "done": torch.tensor(terminated or truncated, dtype=torch.bool),
            },
            batch_size=[1],
        )

        return next_tensordict

    def _cal_reward(
        self, prev_gains: np.ndarray, cur_gains: np.ndarray, out_of_bounds: float
    ) -> float:

        adjusted_gain = np.mean(cur_gains)
        adjusted_gain = np.where(
            adjusted_gain < -100.0,
            (adjusted_gain + 100.0) / 10.0,
            (adjusted_gain + 100.0) / 10.0 + 1.0,
        )
        gain_diff = np.mean(cur_gains - prev_gains)

        reward = float(adjusted_gain + 0.1 * gain_diff - 0.3 * out_of_bounds) / 2.0

        return reward

    def _blender_step(self, focals: np.ndarray[float]) -> np.ndarray[float]:
        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        source_dir = utils.get_os_dir("SOURCE_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        tmp_dir = utils.get_os_dir("TMP_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)

        focal_path = os.path.join(
            tmp_dir, f"data-{self.log_string}-{self.current_time}-{self.idx}.pkl"
        )
        with open(focal_path, "wb") as f:
            pickle.dump(focals, f)

        default_rt_pos = np.array(self.default_sionna_config["rt_positions"])
        current_rt_pos = np.array(self.rt_pos)
        rt_pos_deltas = current_rt_pos - default_rt_pos
        delta_rt_pos_path = os.path.join(
            tmp_dir, f"rt_pos_deltas-{self.log_string}-{self.current_time}-{self.idx}.pkl"
        )
        with open(delta_rt_pos_path, "wb") as f:
            pickle.dump(rt_pos_deltas, f)

        blender_script = os.path.join(source_dir, "marlis", "blender_script", "bl_data_center.py")
        blender_cmd = [
            blender_app,
            "-b",
            os.path.join(blender_dir, "models", f"{scene_name}.blend"),
            "--python",
            blender_script,
            "--",
            "-s",
            self.sionna_config["scene_name"],
            "--focal_path",
            focal_path,
            "--rt_delta_path",
            delta_rt_pos_path,
            "-o",
            blender_output_dir,
        ]
        bl_output_txt = os.path.join(tmp_dir, "bl_outputs.txt")
        subprocess.run(blender_cmd, check=True, stdout=open(bl_output_txt, "w"))

        with open(focal_path, "rb") as f:
            angles = pickle.load(f)
        angles = np.asarray(angles, dtype=np.float32)
        return angles

    def _run_sionna_dB(self, eval_mode: bool = False) -> np.ndarray[np.complex64, float]:
        path_gains = self._run_sionna(eval_mode=eval_mode)
        path_gain_dBs = utils.linear2dB(path_gains)
        return path_gain_dBs

    def _run_sionna(self, eval_mode: bool = False) -> Tuple[tf.Tensor, np.ndarray]:
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)
        compute_scene_dir = os.path.join(blender_output_dir, "ceiling_idx")
        compute_scene_path = glob.glob(os.path.join(compute_scene_dir, "*.xml"))[0]
        viz_scene_dir = os.path.join(blender_output_dir, "idx")
        viz_scene_path = glob.glob(os.path.join(viz_scene_dir, "*.xml"))[0]

        sig_cmap = SignalCoverageMap(self.sionna_config, compute_scene_path, viz_scene_path)

        if eval_mode:
            coverage_map = sig_cmap.compute_cmap()

            img_dir = os.path.join(
                assets_dir, "images", self.log_string + self.current_time + f"_{self.idx}"
            )
            render_filename = utils.create_filename(img_dir, f"{scene_name}_00000.png")
            sig_cmap.render_to_file(coverage_map, filename=render_filename)

        sig_cmap.free_memory()

        paths = sig_cmap.compute_paths()
        a, tau = paths.cir(normalize_delays=False, out_type="torch")
        h_freq = paths.cfr(
            frequencies=self.frequency,
            normalize=False,
            normalize_delays=True,
            out_type="numpy",
        )
        h_freq = cir_to_ofdm_channel(self.sionna_config["frequency"], a, tau, normalize=False)

        path_gains = tf.reduce_mean(tf.square(tf.abs(h_freq)), axis=(0, 2, 4, 5, 6))

        return path_gains


def compute_rot_angle(pt1: list, pt2: list) -> Tuple[float, float, float]:
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    z = pt2[2] - pt1[2]

    return cartesian2spherical(x, y, z)


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def spherical2cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


from marlis.utils import utils, timer
import sionna.rt
import gc
from typing import Optional
import os
import tensorflow as tf
import math
from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    Camera,
    PathSolver,
    RadioMapSolver,
    DirectivePattern,
    RadioMap,
    Paths,
)


def prepare_scene(config, filename, cam=None):
    scene = load_scene(filename, merge_shapes=True)

    scene.frequency = config["frequency"]
    scene.synthetic_array = config["synthetic_array"]

    if cam is not None:
        scene.add(cam)

    scene.tx_array = PlanarArray(
        num_rows=config["tx_num_rows"],
        num_cols=config["tx_num_cols"],
        vertical_spacing=config["tx_vertical_spacing"],
        horizontal_spacing=config["tx_horizontal_spacing"],
        pattern=config["tx_pattern"],
        polarization=config["tx_polarization"],
    )
    for i, (tx_pos, rt_pos) in enumerate(zip(config["tx_positions"], config["rt_positions"])):
        tx = Transmitter(
            f"tx_{i}", tx_pos, look_at=rt_pos, color=[0.05, 0.05, 0.9], display_radius=0.5
        )
        scene.add(tx)

    scene.rx_array = PlanarArray(
        num_rows=config["rx_num_rows"],
        num_cols=config["rx_num_cols"],
        vertical_spacing=config["rx_vertical_spacing"],
        horizontal_spacing=config["rx_horizontal_spacing"],
        pattern=config["rx_pattern"],
        polarization=config["rx_polarization"],
    )

    for i, (rx_pos, rx_orient) in enumerate(zip(config["rx_positions"], config["rx_orientations"])):
        rx = Receiver(f"rx_{i}", rx_pos, rx_orient, color=[0.99, 0.01, 0.99], display_radius=0.5)
        scene.add(rx)

    rm = sionna.rt.radio_material.RadioMaterial(
        "itu_metal_01", relative_permittivity=1.0, conductivity=1e7
    )
    sionna.rt.scene.Scene().add(rm)

    return scene


def prepare_camera(config):
    cam = Camera(
        "my_cam",
        position=config["cam_position"],
        look_at=config["cam_look_at"],
    )
    cam.look_at(config["cam_look_at"])
    return cam


class SignalCoverageMap:
    def __init__(
        self,
        config,
        compute_scene_path: str,
        viz_scene_path: str,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):

        self.config = config
        self.seed = seed

        self._compute_scene_path = compute_scene_path
        self._viz_scene_path = viz_scene_path

        self._cam = prepare_camera(self.config)

        self.scene = prepare_scene(self.config, self._compute_scene_path, self.cam)
        for rm in self.scene.radio_materials.values():
            rm.scattering_coefficient = 1 / math.sqrt(3)
            rm.scattering_pattern = DirectivePattern(alpha_r=10)

        self.verbose = verbose

    @property
    def cam(self):
        return self._cam

    def compute_cmap(self, **kwargs) -> RadioMap:
        if self.verbose:
            print(f"Computing coverage map for {self._compute_scene_path}")
            with timer.Timer(
                text="Elapsed coverage map time: {:0.4f} seconds\n",
                print_fn=print,
            ):
                cmap = self._compute_cmap(**kwargs)
        else:
            cmap = self._compute_cmap(**kwargs)

        return cmap

    def _compute_cmap(self, **kwargs) -> RadioMap:

        cm_kwargs = dict(
            scene=self.scene,
            max_depth=self.config["cm_max_depth"],
            cell_size=self.config["cm_cell_size"],
            samples_per_tx=self.config["cm_num_samples"],
            diffuse_reflection=self.config["diffuse_reflection"],
        )
        if self.seed:
            cm_kwargs["seed"] = self.seed
        if kwargs:
            cm_kwargs.update(kwargs)
        rm_solver = RadioMapSolver()
        cmap = rm_solver(**cm_kwargs)
        return cmap

    def compute_paths(self, **kwargs) -> Paths:
        if self.verbose:
            print(f"Computing paths for {self._compute_scene_path}")
            with timer.Timer(
                text="Elapsed paths time: {:0.4f} seconds\n",
                print_fn=print,
            ):
                paths = self._compute_paths(**kwargs)
        else:
            paths = self._compute_paths(**kwargs)

        return paths

    def _compute_paths(self, **kwargs) -> Paths:

        paths_kwargs = dict(
            scene=self.scene,
            max_depth=self.config["path_max_depth"],
            samples_per_src=self.config["path_num_samples"],
            diffuse_reflection=self.config["diffraction"],
            synthetic_array=self.config["synthetic_array"],
        )
        if self.seed:
            paths_kwargs["seed"] = self.seed
        if kwargs:
            paths_kwargs.update(kwargs)
        p_solver = PathSolver()
        paths = p_solver(**paths_kwargs)
        return paths

    def compute_render(self, cmap_enabled: bool = False, paths_enabled: bool = False) -> None:

        cm = self.compute_cmap() if cmap_enabled else None
        paths = self.compute_paths() if paths_enabled else None

        scene = prepare_scene(self.config, self._viz_scene_path, self.cam)

        img_dir = utils.get_image_dir(self.config)
        render_filename = utils.create_filename(
            img_dir, f"{self.config['mitsuba_filename']}_00000.png"
        )
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            coverage_map=cm,
            cm_vmin=self.config["cm_vmin"],
            cm_vmax=self.config["cm_vmax"],
            resolution=self.config["resolution"],
            show_devices=True,
        )
        scene.render_to_file(**render_config)

    def render_to_file(
        self,
        coverage_map: RadioMap = None,
        paths: Paths = None,
        filename: Optional[str] = None,
    ) -> None:
        scene = prepare_scene(self.config, self._viz_scene_path, self.cam)

        if filename is None:
            img_dir = utils.get_image_dir(self.config)
            render_filename = utils.create_filename(
                img_dir, f"{self.config['mitsuba_filename']}_00000.png"
            )
        else:
            render_filename = filename
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            coverage_map=coverage_map,
            cm_vmin=self.config["cm_vmin"],
            cm_vmax=self.config["cm_vmax"],
            resolution=self.config["resolution"],
            show_devices=True,
        )
        scene.render_to_file(**render_config)

    def render(self, coverage_map: RadioMap = None, paths: Paths = None) -> None:
        scene = prepare_scene(self.config, self._viz_scene_path, self.cam)
        render_config = dict(
            camera=self.cam,
            paths=paths,
            coverage_map=coverage_map,
            cm_vmin=self.config["cm_vmin"],
            cm_vmax=self.config["cm_vmax"],
            resolution=self.config["resolution"],
            show_devices=True,
        )
        scene.render(**render_config)

    def get_path_gain_slow(self, coverage_map: RadioMap) -> float:

        coverage_map_tensor = coverage_map.as_tensor()
        coverage_map_centers = coverage_map.cell_centers
        rx_position = self.config["rx_position"]
        distances = tf.norm(coverage_map_centers - rx_position, axis=-1)
        min_dist = tf.reduce_min(distances)
        min_ind = tf.where(tf.equal(distances, min_dist))[0]

        path_gain: tf.Tensor = coverage_map_tensor[0, min_ind[0], min_ind[1]]
        path_gain = float(path_gain.numpy())
        return path_gain

    def get_path_gain(self, coverage_map: RadioMap) -> float:
        coverage_map_tensor = coverage_map.as_tensor()
        coverage_map_centers = coverage_map.cell_centers
        rx_positions = tf.convert_to_tensor(self.config["rx_positions"], dtype=tf.float32)
        top_left_pos = coverage_map_centers[0, 0, 0:2] - (coverage_map.cell_size / 2)
        path_gains = []
        for rx_position in rx_positions:
            x_distance_to_top_left = rx_position[0] - top_left_pos[0]
            y_distance_to_top_left = rx_position[1] - top_left_pos[1]

            ind_y = int(y_distance_to_top_left / coverage_map.cell_size[1])
            ind_x = int(x_distance_to_top_left / coverage_map.cell_size[0])

            path_gain: tf.Tensor = coverage_map_tensor[0, ind_y, ind_x]
            path_gain = float(path_gain.numpy())
            path_gains.append(path_gain)
        return path_gains

    def get_viz_scene(self) -> sionna.rt.Scene:
        return self.scene

    def free_memory(self) -> None:
        tf.keras.backend.clear_session()
        gc.collect()

        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.experimental.reset_memory_stats("GPU:0")
