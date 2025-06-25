from torchrl.data import (
    Bounded,
    Composite,
    Unbounded,
    UnboundedContinuous,
    BoundedContinuous,
    Categorical,
)
from torchrl.envs import EnvBase
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from rs.envs.engine import AutoRestartManager, SignalCoverage
import copy
import numpy as np
from typing import Optional
import time
import queue
import torch
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

"""
tensordict:
{
    "rf_positions",
    "rx_positions",
    "focal_points",
    
    "action": delta_focal_points,
    
    "reward": reward,
    "done": done,
    
}
"""


class Classroom2UE(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(
        self,
        sionna_config,
        seed=None,
        device="cpu",
        *,
        num_runs_before_restart=10,
        eval_mode=False,
    ):

        super().__init__(device=device, batch_size=[1])

        torch.multiprocessing.set_start_method("forkserver", force=True)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.default_sionna_config = copy.deepcopy(sionna_config)
        self.num_runs_before_restart = num_runs_before_restart
        self.eval_mode = eval_mode

        # devices
        rx_positions = torch.tensor(
            sionna_config["rx_positions"], dtype=torch.float32, device=device
        )
        rx_positions = rx_positions.unsqueeze(0)
        self.rx_position_shape = rx_positions.shape
        self.num_rx = len(sionna_config["rx_positions"])

        # Init focal points
        self.num_rf = len(sionna_config["rf_positions"])
        self.n_agents = self.num_rf
        self.init_focals = torch.tensor(
            [[0.0, 0.0, 1.5] for _ in range(self.num_rf)], dtype=torch.float32, device=device
        )
        self.init_focals = self.init_focals.unsqueeze(0)
        self.focal_low = torch.tensor([[[-10.0, -10.0, -4], [-6.5, -10.0, -4.0]]], device=device)
        self.focal_high = torch.tensor([[[6.5, 10.0, 5.0], [10.0, 10.0, 5.0]]], device=device)

        # observations: focal points, rx_positions, rf_positions
        # actions: delta_focal_points

        # flatten all the tensors for observation
        rf_positions = torch.tensor(
            sionna_config["rf_positions"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        tmp_observation = torch.cat([rx_positions, rf_positions, self.init_focals], dim=-1)
        self.observation_shape = tmp_observation.shape
        rx_low = torch.ones_like(rx_positions, device=device) * (-100)
        rx_high = torch.ones_like(rx_positions, device=device) * 100
        rf_low = torch.ones_like(rf_positions, device=device) * (-100)
        rf_high = torch.ones_like(rf_positions, device=device) * 100

        self.observation_low = torch.cat([rx_low, rf_low, self.focal_low], dim=-1)
        self.observation_high = torch.cat([rx_high, rf_high, self.focal_high], dim=-1)

        self.focals = None
        self.rf_positions = None
        self.rx_positions = None
        self.mgr = None
        self._make_spec()

        self.rx_polygon_coords = [
            [(-4.0, 2.0), (-4.0, -5.5), (2.1, 2.1)],
            [(4.0, 2.0), (4.0, -5.5), (-2.1, 2.1)],
        ]
        self.tx_positions = torch.tensor(
            sionna_config["tx_positions"], dtype=torch.float32, device=device
        )
        self.tx_positions = self.tx_positions.unsqueeze(0)

    def _get_ob(self, tensordict: TensorDictBase) -> TensorDictBase:

        rf_positions = tensordict["agents", "rf_positions"]
        rx_positions = tensordict["agents", "rx_positions"]
        focals = tensordict["agents", "focals"]
        observation = torch.cat([rx_positions, rf_positions, focals], dim=-1)
        tensordict["agents", "observation"] = observation

    def _generate_random_point_in_polygon(self, polygon: Polygon) -> Point:
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            random_point = Point(
                self.np_rng.uniform(min_x, max_x), self.np_rng.uniform(min_y, max_y)
            )
            if polygon.contains(random_point):
                return random_point

    def _prepare_rx_positions(self) -> list:
        """
        Prepare receiver positions based on the defined polygons.
        This function generates random points within the polygons defined in self.rx_polygon_coords.
        """
        rx_pos = []
        for polygon_coords in self.rx_polygon_coords:
            polygon = Polygon(polygon_coords)
            pt = self._generate_random_point_in_polygon(polygon)
            if len(pt.coords[0]) == 2:
                pt = Point(pt.x, pt.y, 1.5)
            pt = [float(coord) for coord in pt.coords[0]]
            rx_pos.append(pt)
        return rx_pos

    def _generate_moved_rx_positions(self, pos: np.ndarray, polygon: Polygon) -> Point:
        """
        Generate receiver positions by moving the original positions slightly within a defined range.
        This function modifies the receiver positions by moving them within a circle of radius 0.2m.
        """
        r = 0.3  # radius of the circle to move the position
        while True:
            random_angle = self.np_rng.uniform(0, 2 * np.pi)
            x = pos[0] + r * np.cos(random_angle)
            y = pos[1] + r * np.sin(random_angle)
            point = Point(x, y)
            if polygon.contains(point):
                return point

    def _move_rx_positions(self) -> list:
        """
        Move the receiver positions slightly within a defined range.
        This function modifies the receiver positions by moving them within a circle of radius 0.2m.
        """
        moved_rx_positions = []
        for idx, pos in enumerate(self.rx_positions.squeeze(0).tolist()):
            # move the position in 0.2m range using a circle with radius 0.2m
            polygon = Polygon(self.rx_polygon_coords[idx])
            pt = self._generate_moved_rx_positions(pos, polygon)
            if len(pt.coords[0]) == 2:
                pt = Point(pt.x, pt.y, 1.5)
            pt = [float(coord) for coord in pt.coords[0]]
            moved_rx_positions.append(pt)
        return moved_rx_positions

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:

        sionna_config = copy.deepcopy(self.default_sionna_config)
        rf_positions = sionna_config["rf_positions"]
        rf_positions = torch.tensor(rf_positions, dtype=torch.float32, device=self.device)
        rf_positions = rf_positions.unsqueeze(0)
        self.rf_positions = rf_positions

        if self.focals is None or not self.eval_mode:
            rx_positions = self._prepare_rx_positions()
        else:
            rx_positions = self._move_rx_positions()
        sionna_config["rx_positions"] = rx_positions
        rx_positions = torch.tensor(rx_positions, dtype=torch.float32, device=self.device)
        rx_positions = rx_positions.unsqueeze(0)
        self.rx_positions = rx_positions

        # Initialize the environment using the Sionna configuration
        if self.focals is None:
            self._init_manager(sionna_config)
        else:
            task_counter = self.mgr.task_counter
            self.mgr.shutdown()
            self._init_manager(
                sionna_config,
                task_counter=task_counter,
            )

        # Focal points
        if self.focals is None or not self.eval_mode:
            # Randomly initialize focal points
            delta_focals = torch.randn_like(self.init_focals)
            delta_focals[..., :2] = delta_focals[..., :2] * 1.5  # Scale x and y by 1.5
            focals = self.init_focals + delta_focals
        else:
            focals = self.focals
        self.focals = torch.clamp(focals, self.focal_low, self.focal_high)

        self.cur_rss = self._get_rss(self.focals)
        self.prev_rss = self.cur_rss.clone().detach()

        out = {
            "agents": {
                "rf_positions": self.rf_positions,
                "rx_positions": self.rx_positions,
                "focals": self.focals.clone().detach(),
                "prev_rss": self.prev_rss,
                "cur_rss": self.cur_rss,
            }
        }
        out = TensorDict(out, device=self.device, batch_size=1)
        self._get_ob(out)

        return out

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Perform a step in the environment."""

        # tensordict  contains the current state of the environment and the action taken
        # by the agent.
        delta_focals = tensordict["agents", "action"]
        self.focals = self.focals + delta_focals

        # if the z values of any focal point is at the boundary of low and high, terminated = True
        truncated = torch.any(
            torch.logical_or(self.focals < self.focal_low, self.focals > self.focal_high)
        )
        truncated = truncated.unsqueeze(0)
        done = truncated.clone()
        # set terminated to ba always False with same shape as truncated
        terminated = torch.zeros_like(truncated, dtype=torch.bool, device=self.device)

        self.focals = torch.clamp(self.focals, self.focal_low, self.focal_high)
        # terminated = torch.tensor([False], dtype=torch.bool, device=self.device).unsqueeze(0)

        # Get rss from the simulation
        # ` TODO: prev_rss and cur_rss may need to be normalized from the SimulationWorker
        # // TODO: The rss now is not normalized
        # ` TODO: Now it is normalized!!
        self.prev_rss = self.cur_rss
        self.cur_rss = self._get_rss(self.focals)
        reward = self._calculate_reward(self.cur_rss, self.prev_rss)

        out = {
            "agents": {
                "rf_positions": self.rf_positions,
                "rx_positions": self.rx_positions,
                "focals": self.focals.clone().detach(),
                "prev_rss": self.prev_rss,
                "cur_rss": self.cur_rss,
                "reward": reward,
            },
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
        out = TensorDict(out, device=self.device, batch_size=1)
        self._get_ob(out)

        return out

    def _get_rss(self, focals: torch.Tensor) -> torch.Tensor:

        try:
            # combine tx_positions and focals
            tx_focals = torch.cat([self.tx_positions, focals], dim=-1)
            tx_focals = tx_focals.detach().cpu().numpy()
            self.mgr.run_simulation((tx_focals[0],))
            res = None
            while res is None:
                try:
                    res = self.mgr.get_result(timeout=10)
                except queue.Empty:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    print("KeyboardInterrupt: shutting down")
                    self.mgr.shutdown()
                    raise
                except Exception as e:
                    print(f"Exception: {e}")
                    self.mgr.shutdown()
                    raise
            rss = torch.tensor(res[1], dtype=torch.float32, device=self.device)
            rss = rss.unsqueeze(0)
        except Exception as e:
            print(f"Exception: {e}")
            self.mgr.shutdown()
            raise e
        return rss

    def _calculate_reward(self, cur_rss, prev_rss):
        """Calculate the reward based on the current and previous rss."""
        # Reward is the difference between current and previous rss

        cur_rss = copy.deepcopy(cur_rss) + 20
        prev_rss = copy.deepcopy(prev_rss) + 20

        w1 = 1.0
        # Get the diagonal elements of the cur_rss tensor and put in a list rfs
        rfs = [cur_rss[:, i : i + 1, i : i + 1] for i in range(self.num_rf)]
        rfs = torch.concat(rfs, dim=1)

        w2 = 0.1
        prev_rfs = [prev_rss[:, i : i + 1, i : i + 1] for i in range(self.num_rf)]
        prev_rfs = torch.concat(prev_rfs, dim=1)
        rfs_diff = rfs - prev_rfs

        reward = 1 / 30 * (w1 * rfs + w2 * rfs_diff)

        return reward

    def _make_spec(self):
        # Under the hood, this will populate self.output_spec["observation"]

        # print(f"focal_low: {focal_low.shape}")
        # print((1, *self.init_focals.shape))
        self.observation_spec = Composite(
            agents=Composite(
                observation=Bounded(
                    low=self.observation_low,
                    high=self.observation_high,
                    shape=self.observation_shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
                rx_positions=Bounded(
                    low=-100,
                    high=100,
                    shape=self.init_focals.shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
                focals=Bounded(
                    low=self.focal_low,
                    high=self.focal_high,
                    shape=self.init_focals.shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
                rf_positions=Bounded(
                    low=-100,
                    high=100,
                    shape=self.init_focals.shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
                prev_rss=UnboundedContinuous(
                    shape=(1, self.num_rf, self.num_rx), dtype=torch.float32, device=self.device
                ),
                cur_rss=UnboundedContinuous(
                    shape=(1, self.num_rf, self.num_rx), dtype=torch.float32, device=self.device
                ),
                shape=(1,),
            ),
            shape=(1,),
            device=self.device,
        )
        # self.state_spec = self.observation_spec.clone()
        self.action_spec = Composite(
            agents=Composite(
                action=BoundedContinuous(
                    low=-0.5,
                    high=0.5,
                    shape=self.init_focals.shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
                shape=(1,),
            ),
            shape=(1,),
            device=self.device,
        )

        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(shape=(1, self.n_agents, 1), device=self.device),
                shape=(1,),
            ),
            shape=(1,),
            device=self.device,
        )

        self.done_spec = Composite(
            done=Categorical(
                n=2,
                shape=(1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Categorical(
                n=2,
                shape=(1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(1,),
            device=self.device,
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _init_manager(self, sionna_config, task_counter=0):
        """
        Initialize the manager for the environment.
        This is called when the environment is created.
        """
        if self.mgr is not None:
            self.mgr.shutdown()
            self.mgr = None
        self.mgr = AutoRestartManager(sionna_config, self.num_runs_before_restart, task_counter)

    def close(self):
        """Close the environment."""
        # release the cuda
        if self.mgr is not None:
            self.mgr.shutdown()
            self.mgr = None
        # clear CUDA
        torch.cuda.empty_cache()
        super().close()


def _add_batch_dim_(tensordict: TensorDictBase, device: str) -> TensorDictBase:
    """
    Add batch dimension to the tensordict.
    This is useful for environments that expect a batch dimension.
    """
    for key in tensordict.keys():
        if "action" not in key and "reward" not in key:
            if isinstance(tensordict[key], torch.Tensor):
                tensordict[key] = tensordict[key].unsqueeze(0)
            else:
                tensordict[key] = torch.tensor(tensordict[key], device=device).unsqueeze(0)
