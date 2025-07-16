# simulation_worker_linux.py
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import multiprocessing as mp
import tensorflow as tf
import time
import queue
import numpy as np
import mitsuba as mi
from rs.utils import utils
import traceback
import pickle
import subprocess
import tempfile
import tensorflow as tf
import math
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    Camera,
    PathSolver,
    ITURadioMaterial,
    SceneObject,
    PathSolver,
    RadioMapSolver,
    DirectivePattern,
    Paths,
    RadioMap,
    Scene,
)
import sionna.rt
from rs.utils import utils
from typing import Optional, Tuple

tf.get_logger().setLevel("ERROR")


# --------------------------
# Persistent Worker Process
# --------------------------
class SimulationWorker(mp.Process):
    def __init__(self, sionna_config: dict, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__()
        self.sionna_config = sionna_config
        self.task_queue = task_queue
        self.result_queue = result_queue

        # Rendering if given in sionna_config
        self.rendering = sionna_config.get("rendering", False)
        self.image_dir = sionna_config.get("image_dir", None)
        if self.rendering and self.image_dir is None:
            raise ValueError("Rendering is enabled, but no image directory is specified.")
        if self.rendering:
            utils.mkdir_not_exists(self.image_dir)

        # Set up paths for blender
        self.blender_app = self.sionna_config["blender_app"]
        self.scene_name = self.sionna_config["scene_name"]

        self.tmp_dir = os.path.join(tempfile.gettempdir(), self.scene_name)
        utils.mkdir_not_exists(self.tmp_dir)
        self.focals_path = os.path.join(self.tmp_dir, f"focals.pkl")

        blender_dir = self.sionna_config["blender_dir"]
        self.blender_model = os.path.join(blender_dir, "models", f"{self.scene_name}.blend")

        self.num_rx = len(sionna_config["rx_positions"])
        self.num_rf = len(sionna_config["rf_positions"])

    def run(self):
        """TF 2.x simulation with forkserver-compatible initialization"""
        # Configure GPU after fork
        self._configure_gpu()

        # Process tasks until shutdown
        while True:
            try:
                task_id, params = self.task_queue.get(timeout=10)
                # print(f"\nProcessing task {task_id} with params: {params}")

                if task_id == "SHUTDOWN":
                    break

                try:
                    # 1) get scene from blender
                    focals = params[0]
                    self._blender_step(focals)

                    # 2) run simulation
                    sig_map = SignalCoverage(self.sionna_config, 0)
                    params = (sig_map, task_id)
                    result = self._run_simulation(*params)

                    # 3) return the result to the main process
                    # Attempt to send the result to the main process
                    self.result_queue.put((task_id, result))
                    del sig_map  # Delete the simulation object to free memory
                    self._clear_memory()  # Clear GPU memory

                except TypeError as e:
                    # Handle serialization errors
                    error_message = (
                        f"Serialization error in _run_simulation: {str(e)}\n"
                        f"Ensure that the result is serializable.\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                    self.result_queue.put((task_id, RuntimeError(error_message)))
                    break  # Stop the worker process on error
                except Exception as e:
                    # Capture the traceback and send it to the main process
                    error_message = f"Error in _run_simulation: {str(e)}\n"
                    error_message += "Traceback:\n" + traceback.format_exc()
                    self.result_queue.put((task_id, RuntimeError(error_message)))
                    break  # Stop the worker process on error

            except queue.Empty:
                continue

            except Exception as e:
                # Capture the traceback and send it to the main process
                error_message = f"Error in worker process: {str(e)}\n"
                error_message += "Traceback:\n" + traceback.format_exc()
                self.result_queue.put((None, RuntimeError(error_message)))
                break  # Stop the worker process on error

    def _configure_gpu(self):
        """TF 2.x GPU configuration (must be called in worker)"""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Avoid memory fragmentation
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Config must be set before GPU initialization
                raise RuntimeError("GPU configuration must be set before GPU initialization")

    def _run_simulation(self, sig_map: "SignalCoverage", task_id: int) -> np.ndarray:
        """Leaky simulation of Sionna"""
        rm = sig_map.compute_cmap()
        rm_rss = rm.rss
        rx_cell_indices = rm.rx_cell_indices

        rssis = np.zeros((self.num_rf, self.num_rx))
        for rx_idx in range(rx_cell_indices.shape[1]):
            rssi = rm_rss[:, rx_cell_indices[1][rx_idx], rx_cell_indices[0][rx_idx]]
            rssis[:, rx_idx] = rssi

        if self.rendering:
            filename = os.path.join(self.image_dir, f"output_{task_id:05d}.png")
            sig_map.render_to_file(radio_map=rm, filename=filename)

        return rssis

    def _blender_step(self, focals: np.ndarray[float]) -> np.ndarray[float]:
        """
        Step the environment using Blender.

        If action is not given, the environment stays the same with the given angles.
        """

        # 1) get the path to the xml file
        xml_dir = self.sionna_config["xml_dir"]

        # 2) pickle focals so that blender app can read it
        with open(self.focals_path, "wb") as f:
            pickle.dump(focals, f)

        # 3) prepare blender script and arguments
        blender_cmd = [self.blender_app, "-b", self.blender_model]
        python_cmd = ["--python", self.sionna_config["blender_script"]]
        arg_cmd = ["--", "-s", self.scene_name, "--focals_path", self.focals_path, "-o", xml_dir]
        blender_cmd.extend(python_cmd)
        blender_cmd.extend(arg_cmd)

        # 4) run blender app
        # bl_output_txt = os.path.join(self.tmp_dir, "bl_outputs.txt")
        subprocess.run(blender_cmd, check=True, stdout=subprocess.DEVNULL)
        # subprocess.run(blender_cmd, check=True)

    def _clear_memory(self):
        """Clear GPU memory"""
        tf.keras.backend.clear_session()


# --------------------------
# Main Process Controller
# --------------------------
class SimulationManager:
    def __init__(self, sionna_config, task_counter: int = 0):
        self._setup_forkserver()
        self.sionna_config = sionna_config
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker = SimulationWorker(self.sionna_config, self.task_queue, self.result_queue)
        self.worker.start()
        self.task_counter = task_counter

    def _setup_forkserver(self):
        """Linux-specific forkserver configuration"""
        try:
            mp.set_start_method("forkserver", force=True)
            ctx = mp.get_context("forkserver")
            ctx.set_forkserver_preload(["tensorflow", "_configure_gpu"])
        except RuntimeError:
            pass

    def run_simulation(self, params=None):
        """Public interface to submit simulations"""
        self.task_queue.put((self.task_counter, params))
        self.task_counter += 1
        return self.task_counter

    def get_result(self, timeout=5):
        """Non-blocking result retrieval"""
        try:
            task_id, result = self.result_queue.get(timeout=timeout)
            if isinstance(result, Exception):
                # Propagate the exception to the main process
                raise result
            return task_id, result
        except queue.Empty:
            return None

    def shutdown(self):
        """Clean termination sequence"""
        try:
            self._clear_memory()  # Clear GPU memory
            # Send shutdown signal to the worker
            if not self.task_queue._closed:
                self.task_queue.put(("SHUTDOWN", None))
            # Wait for the worker to terminate
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=3)
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            # Ensure queues are closed
            if not self.task_queue._closed:
                self.task_queue.close()
            if not self.result_queue._closed:
                self.result_queue.close()
            self._clear_memory()  # Clear GPU memory

    def _clear_memory(self):
        """Clear GPU memory"""
        tf.keras.backend.clear_session()


class AutoRestartManager(SimulationManager):
    def __init__(self, sionna_config: str, max_sims: int = 50, task_counter: int = 0):
        super().__init__(sionna_config, task_counter)
        self.sim_count = 0
        self.max_sims = max_sims

    def run_simulation(self, params=None):
        if self.sim_count >= self.max_sims:
            self._restart_worker()
        self.sim_count += 1
        return super().run_simulation(params)

    def _restart_worker(self):
        """Restart the worker process"""
        self.shutdown()
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker = SimulationWorker(self.sionna_config, self.task_queue, self.result_queue)
        self.worker.start()
        self.sim_count = 0


class SignalCoverage:
    def __init__(self, sionna_config: dict, seed: Optional[int] = None):

        self.sionna_config = sionna_config
        self.seed = seed
        self.image_dir = self.sionna_config["image_dir"]

        self.cam = None
        self.__prepare_camera()

        self.compute_scene_path = self.sionna_config["compute_scene_path"]
        self.compute_scene = load_scene(self.compute_scene_path, merge_shapes=True)
        self.__prepare_radio_devices(self.compute_scene)

        self.rendering = sionna_config.get("rendering", False)
        if self.rendering:
            self.viz_scene_path = self.sionna_config["viz_scene_path"]
            self.viz_scene = load_scene(self.viz_scene_path, merge_shapes=True)
            self.__prepare_radio_devices(self.viz_scene)

        self.rx_pos = np.array(self.sionna_config["rx_positions"], dtype=np.float32)
        self.rf_pos = np.array(self.sionna_config["rf_positions"], dtype=np.float32)
        self.tx_pos = np.array(self.sionna_config["tx_positions"], dtype=np.float32)

        self.num_rx = len(self.rx_pos)
        self.num_rf = len(self.rf_pos)
        self.num_tx = len(self.tx_pos)

    def __prepare_radio_devices(self, scene: Scene):
        # in Hz; implicitly updates RadioMaterials
        scene.frequency = self.sionna_config["frequency"]

        # Device Setup
        scene.tx_array = PlanarArray(
            num_rows=self.sionna_config["tx_num_rows"],
            num_cols=self.sionna_config["tx_num_cols"],
            vertical_spacing=self.sionna_config["tx_vertical_spacing"],
            horizontal_spacing=self.sionna_config["tx_horizontal_spacing"],
            pattern=self.sionna_config["tx_pattern"],
            polarization=self.sionna_config["tx_polarization"],
        )
        for i, (tx_pos, rf_pos) in enumerate(
            zip(self.sionna_config["tx_positions"], self.sionna_config["rf_positions"])
        ):
            tx = Transmitter(
                name=f"tx_{i}",
                position=tx_pos,
                look_at=rf_pos,
                power_dbm=self.sionna_config["tx_power_dbm"],
                color=[0.05, 0.05, 0.9],
                display_radius=0.5,
            )
            scene.add(tx)

        scene.rx_array = PlanarArray(
            num_rows=self.sionna_config["rx_num_rows"],
            num_cols=self.sionna_config["rx_num_cols"],
            vertical_spacing=self.sionna_config["rx_vertical_spacing"],
            horizontal_spacing=self.sionna_config["rx_horizontal_spacing"],
            pattern=self.sionna_config["rx_pattern"],
            polarization=self.sionna_config["rx_polarization"],
        )

        for i, (rx_pos, rx_orient) in enumerate(
            zip(self.sionna_config["rx_positions"], self.sionna_config["rx_orientations"])
        ):
            rx = Receiver(
                name=f"rx_{i}",
                position=rx_pos,
                orientation=rx_orient,
                color=[0.99, 0.01, 0.99],
                display_radius=0.5,
            )
            scene.add(rx)

    def __prepare_camera(self):
        self.cam = Camera(
            position=self.sionna_config["cam_position"],
            look_at=self.sionna_config["cam_look_at"],
        )
        self.cam.look_at(self.sionna_config["cam_look_at"])

    def compute_cmap(self, **kwargs) -> RadioMap:
        cm_kwargs = dict(
            scene=self.compute_scene,
            cell_size=self.sionna_config["rm_cell_size"],
            max_depth=self.sionna_config["rm_max_depth"],
            samples_per_tx=int(self.sionna_config["rm_num_samples"]),
            diffuse_reflection=self.sionna_config["rm_diffuse_reflection"],
            # stop_threshold=self.sionna_config["rm_stop_threshold"],
            rr_depth=self.sionna_config["rm_rr_depth"],
        )
        if self.seed:
            cm_kwargs["seed"] = self.seed
        if kwargs:
            cm_kwargs.update(kwargs)
        rm_solver = RadioMapSolver()
        cmap = rm_solver(**cm_kwargs)
        return cmap

    def compute_paths(self, **kwargs) -> Paths:
        paths_kwargs = dict(
            scene=self.compute_scene,
            max_depth=self.sionna_config["path_max_depth"],
            samples_per_src=int(self.sionna_config["path_num_samples"]),
            diffuse_reflection=self.sionna_config["diffuse_reflection"],
            synthetic_array=self.sionna_config["synthetic_array"],
        )
        if self.seed:
            paths_kwargs["seed"] = self.seed
        if kwargs:
            paths_kwargs.update(kwargs)
        p_solver = PathSolver()
        paths = p_solver(**paths_kwargs)
        return paths

    def render_to_file(
        self, radio_map: RadioMap = None, paths: Paths = None, filename: Optional[str] = None
    ) -> None:
        if not self.rendering:
            raise RuntimeError("Rendering is not enabled in the configuration.")

        if filename is None:
            render_filename = utils.create_filename(
                self.image_dir, f"{self.sionna_config['mitsuba_filename']}_00000.png"
            )
        else:
            render_filename = filename
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            radio_map=radio_map,
            rm_metric="rss",
            rm_vmin=self.sionna_config["rm_vmin"],
            rm_vmax=self.sionna_config["rm_vmax"],
            resolution=self.sionna_config["resolution"],
            show_devices=True,
        )
        self.viz_scene.render_to_file(**render_config)

    def render(self, radio_map: RadioMap = None, paths: Paths = None) -> None:
        if not self.rendering:
            raise RuntimeError("Rendering is not enabled in the configuration.")

        render_config = dict(
            camera=self.cam,
            paths=paths,
            radio_map=radio_map,
            rm_metric="rss",
            rm_vmin=self.sionna_config["rm_vmin"],
            rm_vmax=self.sionna_config["rm_vmax"],
            resolution=self.sionna_config["resolution"],
            show_devices=True,
            rm_show_color_bar=True,
        )
        self.viz_scene.render(**render_config)
