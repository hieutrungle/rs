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
from torch.nn import functional as F
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from rs.modules.agents import allocation
import sionna.rt

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


class Conference2UEAllocation(EnvBase):
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
        self.selected_rx_positions = None
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
        self.n_targets = self.num_rx

        self.distances = torch.zeros(
            (self.n_agents, self.n_targets), dtype=torch.float32, device=device
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

        rf_positions = self.rf_positions
        rx_positions = self.rx_positions
        focals = self.focals
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

    def _calculate_compatibility(self):

        tx_positions = self.tx_positions.squeeze(0)
        rf_positions = self.rf_positions.squeeze(0)
        rx_positions = self.rx_positions.squeeze(0)
        vec_rf_tx = tx_positions - rf_positions  # (n_rf, 3)
        vec_rf_rx = rx_positions.unsqueeze(0) - rf_positions.unsqueeze(1)  # (n_rf, n_rx, 3)

        # Expand vec_rf_tx to (n_rf, n_rx, 3) for broadcasting
        vec_rf_tx_exp = vec_rf_tx.unsqueeze(1).expand(-1, self.n_targets, -1)  # (n_rf, n_rx, 3)

        # Dot product along last dimension
        dot = (vec_rf_tx_exp * vec_rf_rx).sum(dim=-1)  # (n_rf, n_rx)
        norm_tx_rf = vec_rf_tx.norm(dim=-1, keepdim=True)  # (n_rf, 1)
        norm_rf_rx = vec_rf_rx.norm(dim=-1)  # (n_rf, n_rx)
        denom = norm_tx_rf * norm_rf_rx  # (n_rf, n_rx)

        # Avoid division by zero
        cos_angle = dot / (denom + 1e-8)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angles = torch.acos(cos_angle)  # (n_rf, n_rx)
        angles = torch.abs(angles)
        min_val = 0.45
        max_val = 0.8
        angles = (angles - min_val) / (max_val - min_val)

        # compatibility score, shape (n_rf, n_rx)
        # the angles for each rf are close to each other, we need to use a method to make the difference larger. The smaller the angle, the higher the compatibility.
        compat = self.power_law_stretch(angles, gamma=2.0)
        compat = 1.0 / (1.0 + compat)  # Inverse of angle, higher angle means lower compatibility
        return compat

    def _normalize_allocation_states(self):
        """
        Normalize the allocation states (agent and target states) to have zero mean and unit variance.
        This is useful for training neural networks.
        """
        mean = 0.0
        std = 8.0
        self.allocation_agent_states = (self.allocation_agent_states - mean) / std
        self.allocation_target_states = (self.allocation_target_states - mean) / std

    def _get_state(self, tensordict: TensorDict):
        """Get current state representation for all agents"""
        # Create global state representation
        # Allocation/controller observations
        global_allocation_ob = {
            "allocator": {
                "agent_states": self.allocation_agent_states,
                "target_states": self.allocation_target_states,
                # "selected_rx_positions": self.selected_rx_positions,
                # "selected_loc_indices": self.selected_loc_indices.unsqueeze(0),
                # "allocation_logits": self.allocation_logits,
                "compatibility": self.compatibility_matrix.unsqueeze(0),
                "allocation_mask": self.allocation_mask.unsqueeze(0),
            }
        }
        tensordict.update(global_allocation_ob, inplace=True)

        # low-level global observations for agent critics
        rf_positions = self.rf_positions
        rx_positions = self.rx_positions
        focals = self.focals
        global_observation = torch.cat([rx_positions, rf_positions, focals], dim=-1)
        tensordict["global_agent_observation"] = global_observation

        # low-level individual observations for agent actors
        # This is the observation for each agent, which includes its own state and the states of all targets
        # Here we assume that the agent's own state is its rf position and focal
        rf_positions = self.rf_positions.clone().detach()
        rx_positions = self.selected_rx_positions.clone().detach()
        rx_positions = self.rx_positions.clone().detach()
        focals = self.focals.clone().detach()
        observation = torch.cat([rx_positions, rf_positions, focals], dim=-1)
        tensordict["agents", "observation"] = observation

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:

        # Initialize the receiver positions
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

        self.distances = torch.cdist(
            self.rf_positions.squeeze(0), self.rx_positions.squeeze(0), p=2
        )  # (n_rf, n_rx)
        self.factor = torch.pow(self.distances, 2.2)

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

        # Compatibility for allocator
        self.compatibility_matrix = self._calculate_compatibility()
        if self.no_compatibility_scores:
            self.compatibility_matrix = self.compatibility_matrix * 0.0
        self.allocation_agent_states = torch.cat([rf_positions, focals], dim=-1)
        self.allocation_target_states = rx_positions.clone()
        self._normalize_allocation_states()

        # Initialize allocation tracking
        # agent: (rf_x, rf_y, rf_z, fp_x, fp_y, fp_z)
        # target: (rx_x, rx_y, rx_z, angle{tx-tf-rx})
        # compatibility_score ~ (1 / abs(angle))
        if self.no_allocator or self.random_assignment:
            self.selected_loc_indices = torch.randint(
                0, self.num_rx, (self.num_rf,), device=self.device
            )
        else:
            allocator = allocation.GraphAttentionTaskAllocator(
                agent_state_dim=6,
                target_state_dim=3,
                embed_dim=128,
                num_heads=4,
                num_layers=1,
                device=self.device,
            )
            allocator.eval()
            allocator.load_state_dict(torch.load(self.allocator_path, map_location=self.device))
            allocation_state = {
                "agent_states": self.allocation_agent_states,
                "target_states": self.allocation_target_states,
                "compatibility_scores": self.compatibility_matrix.unsqueeze(0),
            }
            with torch.no_grad():
                allocation_outputs = allocator(**allocation_state)

            # Sample actions from allocation probabilities
            self.allocation_logits = allocation_outputs[0]
            allocation_logits = self.allocation_logits.squeeze(0)
            allocation_probs = F.softmax(allocation_logits, dim=-1)
            self.selected_loc_indices = torch.argmax(allocation_probs, dim=-1)
        self.rx_positions = self.rx_positions.to(self.device)
        self.selected_rx_positions = self.rx_positions[0, self.selected_loc_indices, :].unsqueeze(0)

        # if selected_loc_indices are overlapping, we need to penalize the reward
        if len(set(self.selected_loc_indices.tolist())) < self.num_rf:
            self.allocator_reward_const = torch.ones((1, 1), device=self.device) * (-10.0) / 30.0
        else:
            self.allocator_reward_const = torch.ones((1, 1), device=self.device) * 10.0 / 30.0

        self.cur_rss = self._get_rss(self.focals)
        self.prev_rss = self.cur_rss.clone().detach()

        out = {
            "agents": {
                "rf_positions": self.rf_positions.clone().detach(),
                "rx_positions": self.selected_rx_positions.clone().detach(),
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
        allocator_reward = rewards["allocator_reward"]

        # Allocator
        self.allocation_agent_states = torch.cat([self.rf_positions, self.focals], dim=-1)
        self.allocation_target_states = self.rx_positions.clone()
        self._normalize_allocation_states()

        out = {
            "agents": {
                "rf_positions": self.rf_positions.clone().detach(),
                "rx_positions": self.selected_rx_positions.clone().detach(),
                "focals": self.focals.clone().detach(),
                "prev_rss": self.prev_rss,
                "cur_rss": self.cur_rss,
                "reward": agents_reward,
            },
            "allocator": {
                "reward": allocator_reward,
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
        """Calculate the reward based on the current and previous rss.
        Args:
            cur_rss: Current received signal strength (RSS) tensor. (n_rf, n_rx)
            prev_rss: Previous received signal strength (RSS) tensor. (n_rf, n_rx)
        Returns:
            agents_reward: Calculated reward tensor. (n_rf, n_rx)
        """
        # Reward is the difference between current and previous rss

        cur_rss = cur_rss * self.factor.unsqueeze(0)
        prev_rss = prev_rss * self.factor.unsqueeze(0)
        # Convert to dBm
        cur_rss = 10 * torch.log10(cur_rss) + 30.0
        prev_rss = 10 * torch.log10(prev_rss) + 30.0

        # Reward Engineering
        cur_rss = copy.deepcopy(cur_rss)
        prev_rss = copy.deepcopy(prev_rss)
        if self.no_allocator:
            rfs = torch.mean(cur_rss)  # shape: ()
            rfs = rfs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            rfs = torch.repeat_interleave(rfs, self.num_rf, dim=-2)
            prev_rfs = torch.mean(prev_rss)
            prev_rfs = prev_rfs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            prev_rfs = torch.repeat_interleave(prev_rfs, self.num_rf, dim=-2)
        else:
            loc_idx = self.selected_loc_indices  # shape: (num_rf,)
            rf_idx = torch.arange(self.num_rf, device=cur_rss.device)
            # shape: (1, num_rf, 1)
            rfs = cur_rss[:, rf_idx, loc_idx].unsqueeze(-1)
            prev_rfs = prev_rss[:, rf_idx, loc_idx].unsqueeze(-1)
        c = 80
        rfs += c
        prev_rfs += c
        w1 = 1.0
        w2 = 0.1
        rfs_diff = rfs - prev_rfs

        agents_reward = 1 / 30 * (w1 * rfs + w2 * rfs_diff)

        rfs = torch.clamp(rfs, min=0.0)
        allocator_reward = torch.log1p(rfs)  # log(1 + rfs) to avoid log(0)
        allocator_reward = allocator_reward.mean(dim=-2) + self.allocator_reward_const

        return {
            "agents_reward": agents_reward,
            "allocator_reward": allocator_reward,
        }

    def _make_spec(self):
        # Under the hood, this will populate self.output_spec["observation"]

        self.observation_spec = Composite(
            global_agent_observation=Bounded(
                low=self.observation_low,
                high=self.observation_high,
                shape=self.observation_shape,
                dtype=torch.float32,
                device=self.device,
            ),
            allocator=Composite(
                agent_states=Bounded(
                    low=-100,
                    high=100,
                    shape=(1, self.num_rf, 6),  # (rf_x, rf_y, rf_z, fp_x, fp_y, fp_z)
                    dtype=torch.float32,
                    device=self.device,
                ),
                target_states=Bounded(
                    low=-100,
                    high=100,
                    shape=(1, self.num_rx, 3),  # (rx_x, rx_y, rx_z)
                    dtype=torch.float32,
                    device=self.device,
                ),
                # selected_rx_positions=Bounded(
                #     low=-100,
                #     high=100,
                #     shape=(1, self.num_rx, 3),  # (rx_x, rx_y, rx_z)
                #     dtype=torch.float32,
                #     device=self.device,
                # ),
                compatibility=UnboundedContinuous(
                    shape=(1, self.num_rf, self.num_rx),  # compatibility scores
                    dtype=torch.float32,
                    device=self.device,
                ),
                allocation_mask=Categorical(
                    n=2,
                    shape=(1, self.num_rf, self.num_rx),  # allocation mask
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
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
            allocator=Composite(
                reward=Unbounded(shape=(1, 1), dtype=torch.float32, device=self.device),
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
