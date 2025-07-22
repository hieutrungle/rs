import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import functools
import warnings

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["BATCHED_PIPE_TIMEOUT"] = "200"  # to avoid timeout errors in batched pipe
warnings.filterwarnings("ignore", category=UserWarning, module="torchrl")

from torchinfo import summary
from dataclasses import dataclass
import gc
import copy
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pyrallis
from typing import Callable
import torch.multiprocessing

torch.multiprocessing.set_start_method("forkserver", force=True)
from torchrl.envs import ParallelEnv, EnvBase, SerialEnv
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict import TensorDict, TensorDictBase

# Env
from rs.utils import pytorch_utils, utils
from torchrl.envs import (
    Compose,
    RewardSum,
    TransformedEnv,
    StepCounter,
    ObservationNorm,
    DoubleToFloat,
)
from rs.envs import env_ids

# Utils
torch.manual_seed(0)
from torchrl.record.loggers import TensorboardLogger, WandbLogger, Logger
from matplotlib import pyplot as plt
from tqdm import tqdm

set_composite_lp_aggregate(False).set()


@dataclass
class TrainConfig:

    # General arguments
    command: str = "train"  # the command to run
    load_model: str = "-1"  # Model load file name for resume training, "-1" doesn't load
    load_eval_model: str = "-1"  # Model load file name for evaluation, "-1" doesn't load
    load_allocator: str = "-1"  # Allocator load file name for resume training, "-1" doesn't load
    checkpoint_dir: str = "-1"  # the path to save the model
    allocator_replay_buffer_dir: str = "-1"  # the path to save the replay buffer
    load_allocator_replay_buffer: str = "-1"  # the path to load the replay buffer
    source_dir: str = "-1"  # the path to the source code
    verbose: bool = False  # whether to log to console
    seed: int = 10  # seed of the experiment
    eval_seed: int = 111  # seed of the evaluation
    save_interval: int = 100  # the interval to save the model
    start_step: int = 0  # the starting step of the experiment
    track_wandb: bool = True
    use_compile: bool = False  # whether to use torch.dynamo compiler
    image_dir: str = (
        "-1"  # the path to the image directory, if not provided, it will be set to source_dir
    )
    attention_dim: int = 128  # the dimension of the attention mechanism
    attention_heads: int = 4  # the number of attention heads

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 3  # the number of parallel environments
    ep_len: int = 50  # the maximum length of an episode
    eval_ep_len: int = 50  # the maximum length of an episode

    # Sampling
    frames_per_batch: int = 200  # Number of team frames collected per training iteration
    n_iters: int = 500  # Number of sampling and training iterations
    total_episodes: int = 3000  # Total number of episodes to run in the training

    # Training
    num_epochs: int = 40  # Number of optimization steps per training iteration
    minibatch_size: int = 200  # Size of the mini-batches in each optimization step
    lr: float = 2e-4  # Learning rate
    max_grad_norm: float = 1.0  # Maximum norm for the gradients

    # PPO
    clip_epsilon: float = 0.2  # clip value for PPO loss
    gamma: float = 0.985  # discount factor
    lmbda: float = 0.9  # lambda for generalised advantage estimation
    entropy_eps: float = 1e-4  # coefficient of the entropy term in the PPO loss

    # Wandb logging
    wandb_mode: str = "online"  # wandb mode
    project: str = "RS"  # wandb project name
    group: str = "PPO_raw"  # wandb group name
    name: str = "FirstRun"  # wandb run name

    # Ablation Study
    no_compatibility_scores: bool = False  # whether to disable allocator compatibility scores
    random_assignment: bool = False  # whether to use random assignment of users to targets
    no_allocator: bool = False  # whether to disable the allocator

    def __post_init__(self):
        if self.source_dir == "-1":
            raise ValueError("Source dir is required for training")
        if self.checkpoint_dir == "-1":
            raise ValueError("Checkpoints dir is required for training")
        if self.sionna_config_file == "-1":
            raise ValueError("Sionna config file is required for training")
        if self.allocator_replay_buffer_dir == "-1":
            self.allocator_replay_buffer_dir = os.path.join(
                self.checkpoint_dir, "allocator_replay_buffer"
            )
        utils.mkdir_not_exists(self.checkpoint_dir)
        utils.mkdir_not_exists(self.allocator_replay_buffer_dir)

        self.frames_per_batch = self.frames_per_batch * self.num_envs
        # self.total_frames: int = self.frames_per_batch * self.n_iters
        self.allocator_path = os.path.join(self.checkpoint_dir, "allocator.pt")

        total_steps = self.total_episodes * self.ep_len
        n_iters = total_steps // self.frames_per_batch + 1
        self.n_iters = n_iters
        self.total_frames = self.frames_per_batch * self.n_iters

        device = pytorch_utils.init_gpu()
        self.device = device


def wandb_init(config: TrainConfig) -> None:
    key_filename = os.path.join(config.source_dir, "tmp_wandb_api_key.txt")
    with open(key_filename, "r") as f:
        key_api = f.read().strip()
    wandb.login(relogin=True, key=key_api, host="https://api.wandb.ai")
    wandb.init(
        config=config,
        dir=config.checkpoint_dir,
        project=config.project,
        group=config.group,
        name=config.name,
        mode=config.wandb_mode,
    )


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
class Position:
    """Simple 3-D point with Euclidean operations."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def distance_to(self, other: "Position") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    # convenient vector form -------------------------------------------------
    def vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __repr__(self) -> str:  # nicer printing
        return f"Pos({self.x:.2f},{self.y:.2f},{self.z:.2f})"


# --------------------------------------------------------------------------- #
#  GA optimiser (DEAP)
# --------------------------------------------------------------------------- #
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))


# Create a custom NDArray class that properly handles DEAP operations
class NDArrayIndividual(np.ndarray):
    """Custom numpy array class for DEAP individuals"""

    def __new__(cls, input_array, fitness=None):
        obj = np.asarray(input_array).view(cls)
        if fitness is not None:
            obj.fitness = fitness
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # Preserve fitness attribute during array operations
        self.fitness = getattr(obj, "fitness", creator.FitnessMax())


if not hasattr(creator, "Individual"):
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


def make_random_ind(bounds: np.ndarray) -> "creator.Individual":
    """
    bounds: shape (2, n_agents, 3) -> [0, :, :]=low, [1, :, :]=high
    returns: numpy.ndarray (n_agents, 3)
    """
    lows = bounds[0, ...]
    highs = bounds[1, ...]
    sample = np.random.uniform(lows, highs)

    # Create individual with proper fitness initialization
    individual = creator.Individual(sample)
    individual.fitness = creator.FitnessMax()
    return individual


class GAReflectorOptimizer:
    """Evolve the best focal-point (x,y,z) for max RSSI."""

    def __init__(
        self,
        config: TrainConfig,
        bounds: Tuple[tuple, tuple],  # tuple(focal_low, focal_high)
        envs: ParallelEnv,
        population: int = 60,
        generations: int = 120,
        seed: int = 2025,
    ):
        # ---- parameters -----------------------------------------------------
        random.seed(seed)
        np.random.seed(seed)
        self.config = config
        self.low_bnd = np.asarray(bounds[0], dtype=np.float32)
        self.high_bnd = np.asarray(bounds[1], dtype=np.float32)
        self.bounds_np = np.stack([self.low_bnd, self.high_bnd])
        self.pop_size = population
        self.gens = generations
        self.envs = envs
        self.n_agents = self.envs.action_spec.shape[-2]
        self.n_envs = self.envs.action_spec.shape[0]

        # ---- DEAP toolbox ---------------------------------------------------
        self._setup_deap()

    # -----------------------------------------------------------------------
    def _setup_deap(self):
        tb = self.toolbox = base.Toolbox()
        tb.register("individual", functools.partial(make_random_ind, self.bounds_np))
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate", self._eval)
        tb.register("mate", self._blend_cx)
        tb.register("mutate", self._gauss_mut)
        tb.register("select", tools.selTournament, tournsize=3, fit_attr="fitness")
        # tb.register("select", tools.selTournament, tournsize=3)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        for tag in ("mean", "std", "min", "max"):
            getattr(self.stats, "register")(tag, getattr(np, tag))

    # -----------------------------------------------------------------------
    def _eval(self, new_focals: torch.Tensor) -> float:
        """Evaluate a batch of individuals.

        Args:
            new_focals (tensor): Individual to evaluate, containing 9 focal point coordinates (num_envs, 1, n_agents, 3).

        Returns:
            float: The average RSSI value for all users in the environment.
        """
        new_focals = TensorDict(
            {"agents": {"action": new_focals}}, batch_size=(*new_focals.shape[:2],)
        )
        new_focals = new_focals.to(self.envs.device)
        td = self.envs.step(new_focals)
        cur_rss = td.get(("next", "agents", "cur_rss"))
        cur_rss = 10 * torch.log10(cur_rss) + 30
        dim_except_first = tuple(range(1, len(cur_rss.shape)))
        tot = torch.mean(cur_rss, dim=dim_except_first)
        tot = list(tot.detach().cpu().numpy())
        return tot, list(cur_rss[:, 0, 0, :].detach().cpu().numpy())

    # ------------- GA primitives -----------------
    def _blend_cx(self, c1, c2, alpha=0.2):
        # print(f"\nInitial c1: {c1}, c2: {c2}")
        c1 = copy.deepcopy(c1)
        c2 = copy.deepcopy(c2)
        gamma = np.random.uniform(-alpha, 1 + alpha, size=c1.shape)
        inv_g = 1.0 - gamma
        child1 = inv_g * c1 + gamma * c2
        child2 = gamma * c1 + inv_g * c2
        # print(f"Before child1: {child1}, child2: {child2}")
        np.clip(child1, self.low_bnd, self.high_bnd, out=child1)
        np.clip(child2, self.low_bnd, self.high_bnd, out=child2)
        # print(f"After child1: {child1}, child2: {child2}")
        c1[:] = child1
        c2[:] = child2
        # print(f"Final c1: {c1}, c2: {c2}\n")
        return c1, c2

    def _gauss_mut(self, ind, sigma_ratio=0.1, indpb=0.20):
        # print(f"\nBefore mutation: {ind}")
        mutant = copy.deepcopy(ind)
        mask = np.random.rand(*mutant.shape) < indpb
        widths = (self.high_bnd - self.low_bnd) * sigma_ratio
        mutant[mask] += np.random.normal(0.0, widths[mask])
        np.clip(mutant, self.low_bnd, self.high_bnd, out=mutant)
        # print(f"After mutation: {mutant}")
        return (mutant,)

    # -----------------------------------------------------------------------
    def train(self, train_idx: int = 0) -> Tuple["creator.Individual", float]:

        self.pre_train_config()
        pop, fits, rss, td = self.init_population()
        for ind in pop:
            print(f"Individual: {ind}, Fitness: {ind.fitness.values}")

        # evolutionary loop --------------------------------------------------
        cxpb, mutpb = 0.7, 0.2
        for g in range(self.gens):
            # selection → variation
            offspring = list(map(self.toolbox.clone, self.toolbox.select(pop, len(pop))))
            self.perform_crossover_and_mutation(cxpb, mutpb, offspring)

            # re-evaluate new individuals
            self.evaluate_invalid_individuals(rss, offspring)

            pop = list(map(self.toolbox.clone, self.toolbox.select(pop + offspring, len(pop))))
            self.hof.update(pop)
            best = self.hof[0]

            # Record statistics
            rec = self.stats.compile(pop)
            for k, v in rec.items():
                self.hist[k].append(v)

            if g % 1 == 0:
                print(
                    f"G{g:03d}  avg={rec['mean']:.2f};  max={rec['max']:.2f};  best={best.fitness.values[0]:.2f}"
                )

            # Check for improvement and stop if no improvement
            current_best = float("-inf")
            for ind, r in zip(pop, rss):
                if ind.fitness.values[0] > current_best:
                    current_best = ind.fitness.values[0]
                    current_best_rss = r
            if not self.check_improvement(current_best, current_best_rss):
                break

        print(f"Best focal-point: {np.array(best).reshape(-1, 3)}")
        print(f"Best RSSI values: {self.best_rss}  |  RSSI={best.fitness.values[0]:.2f} dBm")
        # save the best individual, rssi, td, and history
        if self.config.checkpoint_dir != "-1":
            save_data = {
                "best_focal_point": np.array(best).reshape(-1, 3),
                "best_rssi": self.best_rss,
                "td": td,
                "history": self.hist,
            }
            save_path = os.path.join(self.config.checkpoint_dir, f"best_focal_point_{train_idx}.pt")
            torch.save(save_data, save_path)
        return best, best.fitness.values[0]

    def evaluate_invalid_individuals(self, rss, offspring):
        invalids = []
        idxs = []
        for idx, ind in enumerate(offspring):
            if not ind.fitness.valid:
                invalids.append(ind)
                idxs.append(idx)

        # invalids = [i for i in offspring if not i.fitness.valid]
        # pad invalids to match multiple of num_envs
        n_invalid = len(invalids)
        if n_invalid % self.n_envs != 0:
            n_pad = self.n_envs - (n_invalid % self.n_envs)
            invalids += [self.toolbox.clone(invalids[0]) for _ in range(n_pad)]
            idxs += [idxs[0] for _ in range(n_pad)]

        invalid_tensor = torch.tensor(list(invalids), dtype=torch.float32, device=self.envs.device)
        invalid_tensor = invalid_tensor.reshape(invalid_tensor.shape[0], 1, self.n_agents, 3)
        invalid_tensor = invalid_tensor.reshape(-1, self.n_envs, *invalid_tensor.shape[1:])

        invalid_fits_rss = list(map(self.toolbox.evaluate, invalid_tensor))
        invalid_fits = []
        invalid_rss = []
        for f, r in invalid_fits_rss:
            invalid_fits.extend(f)
            invalid_rss.extend(r)
        invalid_fits = invalid_fits[:n_invalid]
        invalid_rss = invalid_rss[:n_invalid]

        # # evaluate invalids individuals
        for idx, ind, f, r in zip(idxs, invalids, invalid_fits, invalid_rss):
            ind.fitness.values = (f,)
            rss[idx] = r

    def perform_crossover_and_mutation(self, cxpb, mutpb, offspring):

        # for o in offspring:
        #     print(f"Offspring: {o}, Fitness: {o.fitness.values}")
        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                self.toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # mutation
        for m in offspring:
            if random.random() < mutpb:
                self.toolbox.mutate(m)
                del m.fitness.values

        # for o in offspring:
        #     print(f"Offspring: {o}, Fitness: {o.fitness.values}")

    def init_population(self) -> List[creator.Individual]:
        pop = self.toolbox.population(n=self.pop_size)
        pop_tensor = torch.tensor(copy.deepcopy(pop), dtype=torch.float32, device=self.envs.device)
        # reshape population to match the expected input shape
        pop_tensor = pop_tensor.reshape(-1, self.n_envs, 1, *pop_tensor.shape[1:])
        td = self.envs.reset()
        fits_rss = list(map(self.toolbox.evaluate, pop_tensor))
        fits = []
        rss = []
        for f, r in fits_rss:
            fits.extend(f)
            rss.extend(r)

        fits = list(np.array(fits).flatten())
        for ind, f in zip(pop, fits):
            ind.fitness.values = (f,)
            # print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
        return pop, fits, rss, td

    def pre_train_config(self):

        # ---- tracking -------------------------------------------------------
        self.hist = defaultdict(list)
        self.hof = tools.HallOfFame(3, similar=np.array_equal)

        # monitor the improve in the best fitness value
        self.best_fitness = float("-inf")
        self.best_rss = None
        self.no_improvement = 0
        self.max_no_improvement = 5  # number of generations without improvement before stopping

    def check_improvement(self, current_fitness: float, rss: List[float]) -> bool:
        """Check if the current fitness is better than the best fitness."""
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_rss = rss
            self.no_improvement = 0
        else:
            self.no_improvement += 1

        if self.no_improvement >= self.max_no_improvement:
            print(f"No improvement for {self.max_no_improvement} generations, stopping training.")
            return False
        return True

    # -----------------------------------------------------------------------
    def plot(self):
        if not self.hist["mean"]:
            print("Run train() first.")
            return
        gens = range(len(self.hist["mean"]))
        plt.figure(figsize=(10, 4))
        plt.plot(gens, self.hist["mean"], label="Avg")
        plt.plot(gens, self.hist["max"], label="Best")
        plt.fill_between(
            gens,
            np.array(self.hist["mean"]) - np.array(self.hist["std"]),
            np.array(self.hist["mean"]) + np.array(self.hist["std"]),
            alpha=0.3,
        )
        plt.xlabel("Generation")
        plt.ylabel("RSSI (dBm)")
        plt.title("GA Convergence")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


# class GAReflectorOptimizer:
#     """Evolve the best focal-point (x,y,z) for max RSSI."""

#     def __init__(
#         self,
#         config: TrainConfig,
#         bounds: Tuple[tuple, tuple],  # tuple(focal_low, focal_high)
#         envs: ParallelEnv,
#         population: int = 60,
#         generations: int = 120,
#         seed: int = 2025,
#     ):
#         # ---- parameters -----------------------------------------------------
#         random.seed(seed)
#         np.random.seed(seed)
#         self.config = config
#         self.bnd = bounds
#         self.pop_size = population
#         self.gens = generations
#         self.envs = envs
#         self.n_agents = self.envs.action_spec.shape[-2]
#         self.n_envs = self.envs.action_spec.shape[0]

#         # ---- DEAP toolbox ---------------------------------------------------
#         self._setup_deap()

#     # -----------------------------------------------------------------------
#     def _setup_deap(self):
#         tb = self.toolbox = base.Toolbox()
#         # genes

#         bounds = []
#         for idx in range(self.n_agents):
#             tb.register(f"x{idx}", random.uniform, self.bnd[0][idx][0], self.bnd[1][idx][0])
#             tb.register(f"y{idx}", random.uniform, self.bnd[0][idx][1], self.bnd[1][idx][1])
#             tb.register(f"z{idx}", random.uniform, self.bnd[0][idx][2], self.bnd[1][idx][2])
#             bounds.extend([getattr(tb, f"x{idx}"), getattr(tb, f"y{idx}"), getattr(tb, f"z{idx}")])
#         # individuals & population
#         tb.register("individual", tools.initCycle, creator.Individual, bounds, n=1)
#         tb.register("population", tools.initRepeat, list, tb.individual)
#         # operators
#         tb.register("evaluate", self._eval)
#         tb.register("mate", self._blend_cx)
#         tb.register("mutate", self._gauss_mut)
#         tb.register("select", tools.selTournament, tournsize=3)
#         # stats
#         self.stats = tools.Statistics(lambda ind: ind.fitness.values[0])
#         for tag in ("mean", "std", "min", "max"):
#             getattr(self.stats, "register")(tag, getattr(np, tag))

#     # -----------------------------------------------------------------------
#     def _eval(self, new_focals: torch.Tensor) -> float:
#         """Evaluate a batch of individuals.

#         Args:
#             new_focals (tensor): Individual to evaluate, containing 9 focal point coordinates (num_envs, 1, n_agents, 3).

#         Returns:
#             float: The average RSSI value for all users in the environment.
#         """
#         new_focals = TensorDict(
#             {"agents": {"action": new_focals}}, batch_size=(*new_focals.shape[:2],)
#         )
#         new_focals = new_focals.to(self.envs.device)
#         td = self.envs.step(new_focals)
#         cur_rss = td.get(("next", "agents", "cur_rss"))
#         cur_rss = 10 * torch.log10(cur_rss) + 30
#         dim_except_first = tuple(range(1, len(cur_rss.shape)))
#         tot = torch.mean(cur_rss, dim=dim_except_first)
#         tot = list(tot.detach().cpu().numpy())
#         return tot, list(cur_rss[:, 0, 0, :].detach().cpu().numpy())

#     # BLX-α crossover --------------------------------------------------------
#     def _blend_cx(self, c1, c2, alpha=0.2):
#         # c1, c2 shape: (n_agents, 3)
#         print(f"c1: {c1}, c2: {c2}")
#         for i in range(self.n_agents):
#             for j in range(3):
#                 if random.random() < 0.5:
#                     low, high = sorted((c1[i][j], c2[i][j]))
#                     range_ = high - low
#                     c1[i][j] = random.uniform(low - alpha * range_, high + alpha * range_)
#                     c2[i][j] = random.uniform(low - alpha * range_, high + alpha * range_)
#                     # enforce bounds
#                     c1[i][j] = max(self.bnd[0][i][j], min(self.bnd[1][i][j], c1[i][j]))
#                     c2[i][j] = max(self.bnd[0][i][j], min(self.bnd[1][i][j], c2[i][j]))

#         return c1, c2

#     # bounded Gaussian mutation ---------------------------------------------
#     def _gauss_mut(self, ind, mu=0.0, sigma_ratio=0.1, indpb=0.20):

#         for i in range(self.n_agents):
#             for j in range(3):
#                 if random.random() < indpb:
#                     width = (self.bnd[1][i][j] - self.bnd[0][i][j]) * sigma_ratio
#                     ind[i][j] += random.gauss(mu, width)
#                     ind[i][j] = max(self.bnd[0][i][j], min(self.bnd[1][i][j], ind[i][j]))

#         return (ind,)

#     # -----------------------------------------------------------------------
#     def train(self, train_idx: int = 0) -> Tuple["creator.Individual", float]:

#         self.pre_train_config()
#         pop, fits, rss, td = self.init_population()

#         # evolutionary loop --------------------------------------------------
#         cxpb, mutpb = 0.7, 0.2
#         for g in range(self.gens):
#             # selection → variation
#             offspring = list(map(self.toolbox.clone, self.toolbox.select(pop, len(pop))))
#             self.perform_crossover_and_mutation(cxpb, mutpb, offspring)

#             # re-evaluate new individuals
#             self.evaluate_invalid_individuals(rss, offspring)

#             pop[:] = offspring
#             self.hof.update(pop)
#             best = self.hof[0]

#             # Record statistics
#             rec = self.stats.compile(pop)
#             for k, v in rec.items():
#                 self.hist[k].append(v)

#             if g % 1 == 0:
#                 print(
#                     f"G{g:03d}  avg={rec['mean']:.2f};  max={rec['max']:.2f};  best={best.fitness.values[0]:.2f}"
#                 )

#             # Check for improvement and stop if no improvement
#             current_best = float("-inf")
#             for ind, r in zip(pop, rss):
#                 if ind.fitness.values[0] > current_best:
#                     current_best = ind.fitness.values[0]
#                     current_best_rss = r
#             if not self.check_improvement(current_best, current_best_rss):
#                 break

#         print(f"Best focal-point: {np.array(best).reshape(-1, 3)}")
#         print(f"Best RSSI values: {self.best_rss}  |  RSSI={best.fitness.values[0]:.2f} dBm")
#         # save the best individual, rssi, td, and history
#         if self.config.checkpoint_dir != "-1":
#             save_data = {
#                 "best_focal_point": np.array(best).reshape(-1, 3),
#                 "best_rssi": self.best_rss,
#                 "td": td,
#                 "history": self.hist,
#             }
#             save_path = os.path.join(self.config.checkpoint_dir, f"best_focal_point_{train_idx}.pt")
#             torch.save(save_data, save_path)
#         return best, best.fitness.values[0]

#     def evaluate_invalid_individuals(self, rss, offspring):
#         invalids = []
#         idxs = []
#         for idx, ind in enumerate(offspring):
#             if not ind.fitness.valid:
#                 invalids.append(ind)
#                 idxs.append(idx)

#         # invalids = [i for i in offspring if not i.fitness.valid]
#         # pad invalids to match multiple of num_envs
#         n_invalid = len(invalids)
#         if n_invalid % self.n_envs != 0:
#             n_pad = self.n_envs - (n_invalid % self.n_envs)
#             invalids += [self.toolbox.clone(invalids[0]) for _ in range(n_pad)]
#             idxs += [idxs[0] for _ in range(n_pad)]

#         invalid_tensor = torch.tensor(list(invalids), dtype=torch.float32, device=self.envs.device)
#         invalid_tensor = invalid_tensor.reshape(invalid_tensor.shape[0], 1, -1, 3)
#         invalid_tensor = invalid_tensor.reshape(-1, self.n_envs, *invalid_tensor.shape[1:])

#         invalid_fits_rss = list(map(self.toolbox.evaluate, invalid_tensor))
#         invalid_fits = []
#         invalid_rss = []
#         for f, r in invalid_fits_rss:
#             invalid_fits.extend(f)
#             invalid_rss.extend(r)
#         invalid_fits = invalid_fits[:n_invalid]
#         invalid_rss = invalid_rss[:n_invalid]

#         # fits = np.array(list(map(self.toolbox.evaluate, invalid_tensor))).flatten()
#         # fits = list(fits)[:num_invalid]  # only take the first num_invalid fits
#         # # evaluate invalids individuals
#         for idx, ind, f, r in zip(idxs, invalids, invalid_fits, invalid_rss):
#             ind.fitness.values = (f,)
#             rss[idx] = r

#     def perform_crossover_and_mutation(self, cxpb, mutpb, offspring):
#         # crossover
#         for c1, c2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < cxpb:
#                 self.toolbox.mate(c1, c2)
#                 del c1.fitness.values, c2.fitness.values
#             # mutation
#         for m in offspring:
#             if random.random() < mutpb:
#                 self.toolbox.mutate(m)
#                 del m.fitness.values

#     def init_population(self) -> List[creator.Individual]:
#         td = self.envs.reset()
#         pop = self.toolbox.population(n=self.pop_size)
#         pop_tensor = torch.tensor(list(pop), dtype=torch.float32, device=self.envs.device)
#         pop_tensor = pop_tensor.reshape(self.pop_size, 1, self.n_agents, 3)
#         # reshape population to match the expected input shape
#         pop_tensor = pop_tensor.reshape(-1, self.n_envs, *pop_tensor.shape[1:])
#         fits_rss = list(map(self.toolbox.evaluate, pop_tensor))
#         fits = []
#         rss = []
#         for f, r in fits_rss:
#             fits.extend(f)
#             rss.extend(r)

#         fits = list(np.array(fits).flatten())
#         for i, f in zip(pop, fits):
#             i.fitness.values = (f,)
#         return pop, fits, rss, td

#     def pre_train_config(self):

#         # ---- tracking -------------------------------------------------------
#         self.hist = defaultdict(list)
#         self.hof = tools.HallOfFame(3)

#         # monitor the improve in the best fitness value
#         self.best_fitness = float("-inf")
#         self.best_rss = None
#         self.no_improvement = 0
#         self.max_no_improvement = 5  # number of generations without improvement before stopping

#     def check_improvement(self, current_fitness: float, rss: List[float]) -> bool:
#         """Check if the current fitness is better than the best fitness."""
#         if current_fitness > self.best_fitness:
#             self.best_fitness = current_fitness
#             self.best_rss = rss
#             self.no_improvement = 0
#         else:
#             self.no_improvement += 1

#         if self.no_improvement >= self.max_no_improvement:
#             print(f"No improvement for {self.max_no_improvement} generations, stopping training.")
#             return False
#         return True

#     # -----------------------------------------------------------------------
#     def plot(self):
#         if not self.hist["mean"]:
#             print("Run train() first.")
#             return
#         gens = range(len(self.hist["mean"]))
#         plt.figure(figsize=(10, 4))
#         plt.plot(gens, self.hist["mean"], label="Avg")
#         plt.plot(gens, self.hist["max"], label="Best")
#         plt.fill_between(
#             gens,
#             np.array(self.hist["mean"]) - np.array(self.hist["std"]),
#             np.array(self.hist["mean"]) + np.array(self.hist["std"]),
#             alpha=0.3,
#         )
#         plt.xlabel("Generation")
#         plt.ylabel("RSSI (dBm)")
#         plt.title("GA Convergence")
#         plt.grid()
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


def make_env(config: TrainConfig, idx: int) -> Callable:

    def thunk() -> EnvBase:
        # Load Sionna configuration

        sionna_config = utils.load_config(config.sionna_config_file)
        sionna_config["seed"] = config.seed + idx
        sionna_config["num_runs_before_restart"] = 10
        scene_name = f"{sionna_config['scene_name']}_{idx}"
        sionna_config["scene_name"] = scene_name
        xml_dir = sionna_config["xml_dir"]
        xml_dir = os.path.join(xml_dir, scene_name)
        viz_scene_path = os.path.join(xml_dir, "idx", "scenee.xml")
        compute_scene_path = os.path.join(xml_dir, "ceiling_idx", "scenee.xml")
        sionna_config["xml_dir"] = xml_dir
        sionna_config["viz_scene_path"] = viz_scene_path
        sionna_config["compute_scene_path"] = compute_scene_path

        # image_dir = sionna_config["image_dir"]
        image_dir = os.path.join(config.image_dir, scene_name)
        # image_dir = config.image_dir
        sionna_config["image_dir"] = image_dir

        if config.command.lower() == "train":
            sionna_config["rendering"] = False
        else:
            sionna_config["rendering"] = True

        if config.env_id.lower() not in env_ids:
            raise ValueError(f"Unknown environment id: {config.env_id}")
        env_cls = env_ids[config.env_id.lower()]
        env_args = {
            "sionna_config": sionna_config,
            "allocator_path": config.allocator_path,
            "seed": config.seed,
            "device": config.device,
            "num_runs_before_restart": 20,
            "random_assignment": config.random_assignment,
            "no_allocator": config.no_allocator,
            "no_compatibility_scores": config.no_compatibility_scores,
        }

        if config.command.lower() == "eval":
            env_args["eval_mode"] = True
            env_args["seed"] = config.eval_seed
        env = env_cls(**env_args)

        return env

    return thunk


@pyrallis.wrap()
def main(config: TrainConfig):

    if config.command.lower() == "train":
        print(f"=" * 30 + "Training" + "=" * 30)
    else:
        print(f"=" * 30 + "Evaluation" + "=" * 30)

    utils.log_config(config.__dict__)

    # Reset the torch compiler if needed
    torch.compiler.reset()
    torch.multiprocessing.set_start_method("forkserver", force=True)
    pytorch_utils.init_seed(config.seed)

    # envs = SerialEnv(config.num_envs, [make_env(config, idx) for idx in range(config.num_envs)])
    # check_env_specs(envs)

    envs = ParallelEnv(
        config.num_envs,
        [make_env(config, idx) for idx in range(config.num_envs)],
        mp_start_method="forkserver",
        shared_memory=False,
    )
    ob_spec = envs.observation_spec
    ac_spec = envs.action_spec

    observation_shape = ob_spec["agents", "observation"].shape
    loc = torch.zeros(observation_shape, device=config.device)
    scale = torch.ones(observation_shape, device=config.device) * 8.0

    checkpoint = None
    if config.load_model != "-1":
        checkpoint = torch.load(config.load_model)
        print(f"Loaded checkpoint from {config.load_model}")

    envs = TransformedEnv(
        envs,
        Compose(
            ObservationNorm(
                loc=loc,
                scale=scale,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "observation")],
                standard_normal=True,
            ),
            DoubleToFloat(),
            # StepCounter(max_steps=config.ep_len),
            RewardSum(in_keys=[("agents", "reward")], out_keys=[("agents", "episode_reward")]),
        ),
    )

    # TODO: fix bound of GAReflectorOptimizer
    focal_lows = envs.observation_spec["agents", "focals"].low[0, 0].detach().cpu()
    # focal_lows = torch.ones_like(focal_lows, device=config.device) * (-6.5)
    focal_lows = focal_lows.cpu().numpy()
    focal_lows[..., 2] = 1.0  # z-coordinate should be 0.0
    focal_highs = envs.observation_spec["agents", "focals"].high[0, 0].detach().cpu()
    # focal_highs = torch.ones_like(focal_highs, device=config.device) * 6.5
    focal_highs = focal_highs.cpu().numpy()
    focal_highs[..., 2] = 2.0  # z-coordinate should be 0.0
    bounds = tuple([focal_lows, focal_highs])  # bounds for x, y, z

    try:
        if envs.is_closed:
            envs.start()

        # Train the GA
        ga_reflector_optimizer = GAReflectorOptimizer(
            config=config,
            bounds=bounds,  # Example bounds for x, y, z
            envs=envs,
            population=config.num_envs * 2,
            generations=20,
            seed=config.seed,
        )
        pbar = tqdm(total=10)
        for train_idx in range(10):
            # if config.load_model != "-1":
            #     ga_reflector_optimizer.toolbox.register(
            #         "evaluate", ga_reflector_optimizer._eval, checkpoint["best_focal_point"]
            #     )
            # else:
            #     ga_reflector_optimizer.toolbox.register("evaluate", ga_reflector_optimizer._eval)

            # Train the GA
            start_time = time.perf_counter()
            best_focal_point, best_fitness = ga_reflector_optimizer.train(train_idx)
            end_time = time.perf_counter()
            print(f"Training time: {end_time - start_time:.2f} seconds\n")
            # pbar.set_description(f"best_fitness = {best_fitness}", refresh=False)
            pbar.update()
        # ga_reflector_optimizer.plot()

    except Exception as e:
        print("Environment specs are not correct")
        print(e)
        traceback.print_exc()
    finally:
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()
        if not envs.is_closed:
            envs.close()


if __name__ == "__main__":
    main()
