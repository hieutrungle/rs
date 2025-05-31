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


class TwoAgentDataCenter(EnvBase):
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
            [[0.0, 0.0, 2.0] for _ in range(self.num_rf)], dtype=torch.float32, device=device
        )
        self.init_focals = self.init_focals.unsqueeze(0)
        self.focal_low = torch.tensor([[[-5.0, -7.0, -4], [-8.0, -7.0, -4.0]]], device=device)
        self.focal_high = torch.tensor([[[8.0, 7.0, 5.0], [5.0, 7.0, 5.0]]], device=device)

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

        self.rx_ranges_1 = [
            [(2.55, 2.8), (4.0, 4.5)],
            [(2.55, 2.8), (2.8, 3.3)],
            [(2.55, 2.8), (1.6, 2.1)],
            [(2.55, 2.8), (0.4, 0.9)],
            [(2.55, 2.8), (-0.9, -0.4)],
            [(2.55, 2.8), (-2.1, -1.6)],
            [(2.55, 2.8), (-3.3, -2.8)],
            [(2.55, 2.8), (-4.5, -4.0)],
        ]
        self.rx_ranges_2 = [
            [(-2.8, -2.55), (4.0, 4.5)],
            [(-2.8, -2.55), (2.8, 3.3)],
            [(-2.8, -2.55), (1.6, 2.1)],
            [(-2.8, -2.55), (0.4, 0.9)],
            [(-2.8, -2.55), (-0.9, -0.4)],
            [(-2.8, -2.55), (-2.1, -1.6)],
            [(-2.8, -2.55), (-3.3, -2.8)],
            [(-2.8, -2.55), (-4.5, -4.0)],
        ]

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
        Prepare receiver positions based on the 2 defined ranges.
        """
        rx_pos = []
        for rx_ranges in [self.rx_ranges_1, self.rx_ranges_2]:
            rx_range = self.np_rng.choice(rx_ranges, size=1, replace=False)[0]
            x = self.np_rng.uniform(rx_range[0][0], rx_range[0][1])
            y = self.np_rng.uniform(rx_range[1][0], rx_range[1][1])
            pt = [float(x), float(y), 1.5]
            rx_pos.append(pt)
        return rx_pos

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:

        sionna_config = copy.deepcopy(self.default_sionna_config)

        # Reflector positions
        rf_positions = sionna_config["rf_positions"]
        rf_positions = torch.tensor(rf_positions, dtype=torch.float32, device=self.device)
        rf_positions = rf_positions.unsqueeze(0)
        self.rf_positions = rf_positions

        # RX positions
        rx_positions = sionna_config["rx_positions"]
        rx_positions = self._prepare_rx_positions()
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
        delta_focals = torch.randn_like(self.init_focals) * 1.5
        focals = self.init_focals + delta_focals
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
            self.mgr.run_simulation((focals.detach().cpu().numpy()[0],))
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

        cur_rss = copy.deepcopy(cur_rss) + 40
        prev_rss = copy.deepcopy(prev_rss) + 40

        w1 = 1.2
        rf1 = cur_rss[:, 0:1, 0:1]
        rf2 = cur_rss[:, 1:2, 1:2]
        rfs = torch.cat([rf1, rf2], dim=1)

        w2 = 0.1
        rf1_diff = rf1 - prev_rss[:, 0:1, 0:1]
        rf2_diff = rf2 - prev_rss[:, 1:2, 1:2]
        rfs_diff = torch.cat([rf1_diff, rf2_diff], dim=1)

        reward = 0.3 / self.num_rf * (w1 * rfs + w2 * rfs_diff)

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
