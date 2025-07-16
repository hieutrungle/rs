"""
Only 1 RF each column of RF is an agent. There are many users (rx) in the environment.
Each RF has a focal point that can be moved in 3D space.
The goal is to maximize the received signal strength (RSS) at the receiver positions (rx) by adjusting the focal points of the RFs.
The environment uses Sionna for ray tracing to simulate the RSS.
The environment supports both random assignment of focal points and allocation using a trained allocator.
The environment is designed to be used with reinforcement learning algorithms.
The environment provides observations, actions, rewards, and termination conditions.
The observations include the positions of the RFs, the positions of the receivers, and the focal points.
The actions are the changes in the focal points.
The rewards are based on the RSS at the receiver positions.
The environment supports both training and evaluation modes.
The environment can be reset to a new state, and steps can be taken to update the state.
The environment can be closed to release resources.
This code is part of a reinforcement learning environment for conference-like scenarios with multiple RF agents and users.
It is designed to be used with the Sionna library for ray tracing and signal processing.
It includes functionality for managing the environment, calculating rewards, and handling actions.
It is structured to work with the TorchRL library for reinforcement learning.
This code is intended for use in research and development of multi-agent systems in wireless communication environments.
It is not intended for production use and may require further modifications for specific applications.
This code is licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

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
from rs.envs.engine_ma import AutoRestartManager, SignalCoverage
import copy
import numpy as np
from typing import Optional
import time
import queue
import torch
from torch.nn import functional as F
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from rs.modules.agents import allocation
import sionna.rt

"""
tensordict:
{
    "agent_pos",
    "target_pos",
    "focal_points",
    
    "action": delta_focal_points,
    
    "reward": reward,
    "done": done,
    
}
"""


class Hallway1UEMA(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(
        self,
        sionna_config: dict,
        allocator_path: int,
        seed: int = None,
        device: str = "cpu",
        *,
        random_assignment: bool = False,
        no_allocator: bool = False,
        no_compatibility_scores: bool = False,
        num_runs_before_restart: int = 10,
        eval_mode: bool = False,
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
        self.allocator_path = allocator_path
        self.random_assignment = random_assignment
        self.no_allocator = no_allocator
        self.no_compatibility_scores = no_compatibility_scores

        # Init focal points
        self.init_focals = torch.tensor(
            [[-5.0, 3.0, 1.5] for _ in range(9)], dtype=torch.float32, device=device
        )
        self.init_focals = self.init_focals.unsqueeze(0)
        self.focal_low = torch.tensor([[[-18.0, -6.0, -4.0] for _ in range(9)]], device=device)
        self.focal_high = torch.tensor([[[1.0, 10.0, 5.0] for _ in range(9)]], device=device)

        # Generate symmetric RF positions around 'mid' using 'vector'
        mid = np.array([2.09223, -3.88804, 2.06059])
        next_mid = np.array([1.95788, -4.02239, 1.93559])
        vector = mid - next_mid
        agent_pos = []
        pos = mid.copy()
        vec = -vector
        for i in range(4):
            vec[2] *= -1
            pos = pos - vec
            agent_pos.append(pos)
        agent_pos.reverse()
        pos = mid.copy()
        agent_pos.append(pos)
        vec = vector
        vec[2] *= -1
        for i in range(4):
            vec[2] *= -1
            pos = pos - vec
            agent_pos.append(pos)
        agent_pos = np.array(agent_pos, dtype=np.float32)
        agent_pos = torch.tensor(agent_pos, device=device).unsqueeze(0)

        self.n_agents = int(9)
        self.n_targets = len(sionna_config["rx_positions"])

        # ob: (rx_x, rx_y, rx_z, rf_x, rf_y, rf_z, fp_x, fp_y, fp_z) * num_agents
        tmp_observation = torch.cat([agent_pos.clone(), agent_pos, self.init_focals], dim=-1)
        self.observation_shape = tmp_observation.shape
        rx_low = torch.ones_like(agent_pos, device=device) * (-100)
        rx_high = torch.ones_like(agent_pos, device=device) * 100
        rf_low = torch.ones_like(agent_pos, device=device) * (-100)
        rf_high = torch.ones_like(agent_pos, device=device) * 100
        self.observation_low = torch.cat([rx_low, rf_low, self.focal_low], dim=-1)
        self.observation_high = torch.cat([rx_high, rf_high, self.focal_high], dim=-1)

        self.focals = None
        self.agent_pos = agent_pos
        self.target_pos = None  # n_targets, not real_rx
        self.selected_rx_positions = None
        self.mgr = None
        self._make_spec()

        self.rx_polygon_coords = [
            [(-7.0, -2.4), (-7.0, -4.3), (-14.5, -4.3), (-14.5, -2.4)]
            for _ in range(self.n_targets)
        ]
        self.tx_positions = torch.tensor(
            sionna_config["tx_positions"], dtype=torch.float32, device=device
        )
        self.tx_positions = self.tx_positions.unsqueeze(0)

        self.distances = torch.zeros(
            (self.n_agents, self.n_agents), dtype=torch.float32, device=device
        )
        self.allocation_agent_states = None
        self.allocation_target_states = None
        self.allocation_logits = None
        self.compatibility_matrix = None
        self.allocation_mask = torch.zeros(
            self.n_agents, self.n_targets, dtype=torch.bool, device=device
        )
        self.allocator_reward_const = 0.0

    def _get_ob(self, tensordict: TensorDictBase) -> TensorDictBase:

        agent_pos = self.agent_pos
        target_pos = self.target_pos
        focals = self.focals
        observation = torch.cat([target_pos, agent_pos, focals], dim=-1)
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
        for idx, pos in enumerate(self.target_pos.squeeze(0).tolist()):
            # move the position in 0.2m range using a circle with radius 0.2m
            polygon = Polygon(self.rx_polygon_coords[idx])
            pt = self._generate_moved_rx_positions(pos, polygon)
            if len(pt.coords[0]) == 2:
                pt = Point(pt.x, pt.y, 1.5)
            pt = [float(coord) for coord in pt.coords[0]]
            moved_rx_positions.append(pt)
        return moved_rx_positions

    def power_law_stretch(self, tensor, gamma=3.0):
        """
        Stretches the values in a tensor using a power-law transformation.

        Args:
            tensor: The input PyTorch tensor with values in [0, 1].
            gamma: The gamma value. Values > 1 will stretch the data.

        Returns:
            The transformed tensor.
        """
        # # Ensure gamma is greater than 1 for stretching
        # if gamma <= 1.0:
        #     print("Warning: Gamma should be > 1 to increase separation.")

        return torch.pow(tensor, gamma)

    def _get_state(self, tensordict: TensorDict):
        """Get current state representation for all agents"""
        agent_pos = self.agent_pos.clone().detach()
        target_pos = self.target_pos.clone().detach()
        focals = self.focals.clone().detach()
        observation = torch.cat([target_pos, agent_pos, focals], dim=-1)
        tensordict["agents", "observation"] = observation

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:

        # Initialize the receiver positions
        sionna_config = copy.deepcopy(self.default_sionna_config)
        if self.focals is None or not self.eval_mode:
            rx_positions = self._prepare_rx_positions()
        else:
            rx_positions = self._move_rx_positions()
        sionna_config["rx_positions"] = rx_positions
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
        self.init_agent_focals = self.focals.clone()

        # Assign rx to each agent
        self.selected_loc_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        self.target_pos = torch.tensor(
            [rx_positions[i] for i in self.selected_loc_indices],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # calculate the distances between the RFs and RXs with the same indices # (n_agents)
        self.distances = torch.pairwise_distance(
            self.agent_pos.squeeze(0), self.target_pos.squeeze(0), p=2
        )
        self.distances = self.distances.unsqueeze(0)  # Add batch dimension
        self.factors = torch.pow(self.distances, 2.2)

        self.cur_rss = self._get_rss(self.focals)
        self.prev_rss = self.cur_rss.clone().detach()

        out = {
            "agents": {
                "agent_pos": self.agent_pos.clone().detach(),
                "target_pos": self.target_pos.clone().detach(),
                "focals": self.focals.clone().detach(),
                "prev_rss": self.prev_rss,
                "cur_rss": self.cur_rss,
            }
        }
        out = TensorDict(out, device=self.device, batch_size=1)
        self._get_state(out)

        return out

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Perform a step in the environment."""

        # tensordict  contains the current state of the environment and the action taken
        # by the agent.
        delta_focals = tensordict["agents", "action"] * 0.6  # Scale the action by 0.6
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
        self.prev_rss = self.cur_rss
        self.cur_rss = self._get_rss(self.focals)
        rewards = self._calculate_reward(self.cur_rss, self.prev_rss)
        agents_reward = rewards["agents_reward"]

        out = {
            "agents": {
                "agent_pos": self.agent_pos.clone().detach(),
                "target_pos": self.target_pos.clone().detach(),
                "focals": self.focals.clone().detach(),
                "prev_rss": self.prev_rss,
                "cur_rss": self.cur_rss,
                "reward": agents_reward,
            },
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
        out = TensorDict(out, device=self.device, batch_size=1)
        self._get_state(out)

        return out

    def _get_rss(self, focals: torch.Tensor) -> torch.Tensor:

        try:
            # combine tx_positions and focals
            self.mgr.run_simulation((focals[0].detach().cpu().numpy(),))
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
        """Calculate the reward based on the current and previous rss.
        Args:
            cur_rss: Current received signal strength (RSS) tensor. (n_rf, n_rx)
            prev_rss: Previous received signal strength (RSS) tensor. (n_rf, n_rx)
        Returns:
            agents_reward: Calculated reward tensor. (n_rf, n_rx)
        """

        cur_rss = copy.deepcopy(cur_rss)
        prev_rss = copy.deepcopy(prev_rss)
        cur_rss = torch.cat([cur_rss, cur_rss, cur_rss], dim=-1)
        prev_rss = torch.cat([prev_rss, prev_rss, prev_rss], dim=-1)
        cur_rss = cur_rss * self.factors.unsqueeze(0)
        prev_rss = prev_rss * self.factors.unsqueeze(0)

        # Convert to dBm
        cur_rss = 10 * torch.log10(cur_rss) + 30.0
        prev_rss = 10 * torch.log10(prev_rss) + 30.0

        # Reward Engineering
        # shape: (1, n_agents, 1)
        rfs = cur_rss[0].unsqueeze(-1)
        prev_rfs = prev_rss[0].unsqueeze(-1)
        c = 65
        rfs += c
        prev_rfs += c
        w1 = 1.0
        w2 = 0.1
        rfs_diff = rfs - prev_rfs
        agents_reward = 1 / 30 * (w1 * rfs + w2 * rfs_diff)

        return {"agents_reward": agents_reward}

    def _make_spec(self):
        # Under the hood, this will populate self.output_spec["observation"]

        self.observation_spec = Composite(
            agents=Composite(
                observation=Bounded(
                    low=self.observation_low,
                    high=self.observation_high,
                    shape=self.observation_shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
                target_pos=Bounded(
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
                agent_pos=Bounded(
                    low=-100,
                    high=100,
                    shape=self.init_focals.shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
                prev_rss=UnboundedContinuous(
                    shape=(1, self.n_agents, self.n_targets),
                    dtype=torch.float32,
                    device=self.device,
                ),
                cur_rss=UnboundedContinuous(
                    shape=(1, self.n_agents, self.n_targets),
                    dtype=torch.float32,
                    device=self.device,
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
