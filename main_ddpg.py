import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


from dataclasses import dataclass
import traceback
import numpy as np
import torch
import wandb
import pyrallis
from typing import Callable
import copy
import torch.multiprocessing

torch.multiprocessing.set_start_method("forkserver", force=True)
from torchrl.envs import check_env_specs
import torchrl
from torchrl.envs import ParallelEnv, EnvBase, SerialEnv
from tensordict import TensorDict
import time
import gc

torch.multiprocessing.set_start_method("forkserver", force=True)

import rs
from rs.utils import pytorch_utils, utils
from rs.envs import Classroom

# Torch

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer, TensorDictReplayBuffer

from torchrl.envs import (
    check_env_specs,
    Compose,
    RewardSum,
    TransformedEnv,
    StepCounter,
    ObservationNorm,
    DoubleToFloat,
    set_exploration_type,
)
from torchrl.modules import (
    AdditiveGaussianModule,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
)

from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

from tqdm import tqdm

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
    checkpoint_dir: str = "-1"  # the path to save the model
    replay_buffer_dir: str = "-1"  # the path to save the replay buffer
    load_replay_buffer: str = "-1"  # the path to load the replay buffer
    source_dir: str = "-1"  # the path to the source code
    verbose: bool = False  # whether to log to console
    seed: int = 1  # seed of the experiment
    eval_seed: int = 111  # seed of the evaluation
    save_interval: int = 100  # the interval to save the model
    start_step: int = 0  # the starting step of the experiment
    track_wandb: bool = True
    use_compile: bool = False  # whether to use torch.dynamo compiler

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 3  # the number of parallel environments
    ep_len: int = 50  # the maximum length of an episode
    eval_ep_len: int = 50  # the maximum length of an episode

    # Sampling
    frames_per_batch: int = 256  # Number of team frames collected per sampling iteration
    n_iters: int = 500  # Number of sampling and training iterations

    # Replay buffer
    memory_size: int = 75_000  # The replay buffer of each group can store this many frames

    # Training
    num_epochs: int = 100  # Number of optimization steps per training iteration
    minibatch_size: int = 128  # Size of the mini-batches in each optimization step
    lr: float = 3e-4  # Learning rate
    max_grad_norm: float = 0.8  # Maximum norm for the gradients

    # DDPG
    gamma: float = 0.985  # Discount factor
    polyak_tau: float = 0.05  # Tau for the soft-update of the target network

    # Wandb logging
    wandb_mode: str = "online"  # wandb mode
    project: str = "RS"  # wandb project name
    group: str = "DDPG_raw"  # wandb group name
    name: str = "FirstRun"  # wandb run name

    def __post_init__(self):
        if self.source_dir == "-1":
            raise ValueError("Source dir is required for training")
        if self.checkpoint_dir == "-1":
            raise ValueError("Checkpoints dir is required for training")
        if self.sionna_config_file == "-1":
            raise ValueError("Sionna config file is required for training")
        if self.command.lower() == "train" and self.replay_buffer_dir == "-1":
            raise ValueError("Replay buffer dir is required for training")
        # if self.command.lower() == "eval" and self.load_eval_model == "-1":
        #     raise ValueError("Load eval model is required for evaluation")

        utils.mkdir_not_exists(self.checkpoint_dir)

        self.frames_per_batch = self.frames_per_batch * self.num_envs
        self.total_frames: int = self.frames_per_batch * self.n_iters

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

        image_dir = sionna_config["image_dir"]
        image_dir = os.path.join(image_dir, scene_name)
        sionna_config["image_dir"] = image_dir

        seed = config.seed + idx

        if config.command.lower() == "train":
            sionna_config["rendering"] = False
        else:
            sionna_config["rendering"] = True

        env = Classroom(
            sionna_config,
            seed=seed,
            device=config.device,
            num_runs_before_restart=20,
        )

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
            StepCounter(max_steps=config.ep_len),
            RewardSum(in_keys=[envs.reward_key], out_keys=[("agents", "episode_reward")]),
        ),
    )

    try:
        if envs.is_closed:
            envs.start()
        n_agents = list(envs.n_agents)[0]
        shared_parameters_policy = False
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=ob_spec["agents", "observation"].shape[-1],
                n_agent_outputs=ac_spec.shape[-1],  # n_actions_per_agents
                n_agents=n_agents,
                centralised=False,
                share_params=shared_parameters_policy,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "param")],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=envs.action_spec_unbatched,
            in_keys=[("agents", "param")],
            out_keys=[envs.action_key],
            distribution_class=TanhDelta,
            distribution_kwargs={
                "low": envs.full_action_spec_unbatched[envs.action_key].space.low,
                "high": envs.full_action_spec_unbatched[envs.action_key].space.high,
            },
            return_log_prob=False,
        )

        exploration_policy = TensorDictSequential(
            policy,
            AdditiveGaussianModule(
                spec=policy.spec,
                # Number of frames after which sigma is sigma_end
                annealing_num_steps=config.total_frames // 2,
                action_key=("agents", "action"),
                sigma_init=0.9,  # Initial value of the sigma
                sigma_end=0.1,  # Final value of the sigma
            ),
        )

        # Critics
        share_parameters_critic = False
        maddpg = True
        cat_module = TensorDictModule(
            lambda obs, action: torch.cat([obs, action], dim=-1),
            in_keys=[("agents", "observation"), ("agents", "action")],
            out_keys=[("agents", "obs_action")],
        )

        critic_module = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=ob_spec["agents", "observation"].shape[-1] + ac_spec.shape[-1],
                n_agent_outputs=1,  # 1 value per agent
                n_agents=n_agents,
                centralised=maddpg,
                share_params=share_parameters_critic,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            in_keys=[("agents", "obs_action")],
            out_keys=[("agents", "state_action_value")],
        )
        critic = TensorDictSequential(cat_module, critic_module)

        # Collector
        collector = SyncDataCollector(
            envs,
            exploration_policy,
            device=config.device,
            storing_device=config.device,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.total_frames,
        )

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(config.memory_size, device=config.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=config.minibatch_size,
        )
        if config.device.type != "cpu":
            replay_buffer.append_transform(lambda x: x.to(config.device))

        # Loss
        loss_module = DDPGLoss(
            actor_network=policy,  # Use the non-explorative policies
            value_network=critic,
            delay_value=True,  # Whether to use a target network for the value
            loss_function="l2",
        )
        loss_module.set_keys(
            state_action_value=("agents", "state_action_value"),
            reward=("agents", "reward"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config.gamma)

        optim = torch.optim.Adam(loss_module.parameters(), config.lr)

        if checkpoint:
            print(f"Loading checkpoint from: {config.load_model}")
            policy.load_state_dict(checkpoint["policy"])
            critic.load_state_dict(checkpoint["critic"])
            loss_module.load_state_dict(checkpoint["loss_module"])
            optim.load_state_dict(checkpoint["optimizer"])

        if config.command == "train":
            print("Training...")
            train(
                envs,
                config,
                collector,
                policy,
                exploration_policy,
                critic,
                loss_module,
                optim,
                replay_buffer,
            )
        else:
            print("Evaluation...")
            eval(envs, config, policy)
            print("Evaluation done")

    except Exception as e:
        print("Environment specs are not correct")
        print(e)
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if not envs.is_closed:
            envs.close()


def transform_envs(envs, config: TrainConfig):
    envs = TransformedEnv(
        envs,
        Compose(
            ObservationNorm(
                in_keys=[("agents", "observation")], out_keys=[("agents", "observation")]
            ),
            DoubleToFloat(),
            StepCounter(max_steps=config.ep_len),
            RewardSum(in_keys=[envs.reward_key], out_keys=[("agents", "episode_reward")]),
        ),
    )
    return envs


def train(
    envs: ParallelEnv,
    config: TrainConfig,
    collector: SyncDataCollector,
    policy: TensorDictModule,
    exploration_policy: TensorDictModule,
    critic: TensorDictModule,
    loss_module: DDPGLoss,
    optim: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
):
    target_net_updater = SoftUpdate(loss_module, eps=1 - config.polyak_tau)
    if config.track_wandb:
        wandb_init(config)
    else:
        logger = TensorboardLogger(
            exp_name=f"{config.group}_{config.name}",
            log_dir=config.checkpoint_dir,
        )
        saved_config = config.__dict__.copy()
        saved_config["device"] = str(config.device)
        logger.log_hparams(saved_config)

    pbar = tqdm(total=config.n_iters, desc="episode_reward_mean = 0.0")
    episode_reward_mean_list = []
    for idx, tensordict_data in enumerate(collector):

        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", envs.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", envs.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "truncated"),
            tensordict_data.get(("next", "truncated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", envs.reward_key))),
        )

        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        if idx < 1:
            pbar.update()
            continue

        for _ in range(config.num_epochs):
            for _ in range(config.frames_per_batch // config.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = loss_vals["loss_actor"] + loss_vals["loss_value"]
                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.max_grad_norm)
                optim.step()
                optim.zero_grad()
                target_net_updater.step()

        exploration_policy[1].step(frames=config.frames_per_batch)
        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        done = tensordict_data.get(("next", "done"))
        episode_max_step = tensordict_data.get(("next", "step_count"))[done]
        episode_max_step = episode_max_step.to(torch.float).mean().item()
        episode_reward_mean_list.append(episode_reward_mean)
        logs = {
            "episode_max_step": episode_max_step,
            "episode_reward_mean": episode_reward_mean,
            "loss_value": loss_vals["loss_value"].item(),
            "loss_actor": loss_vals["loss_actor"].item(),
        }
        step = idx * config.frames_per_batch
        if config.track_wandb:
            wandb.log({**logs}, step=step)
        else:
            for key, value in logs.items():
                logger.log_scalar(key, value, step=step)
        torch.save(
            {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "loss_module": loss_module.state_dict(),
                "optimizer": optim.state_dict(),
            },
            os.path.join(config.checkpoint_dir, f"checkpoint_{idx}.pt"),
        )
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        pbar.update()


def eval(envs: ParallelEnv, config: TrainConfig, policy: TensorDictModule):
    with torch.no_grad():
        envs.rollout(
            max_steps=config.eval_ep_len,
            policy=policy,
            # callback=lambda env, _: env.render(),
            auto_cast_to_device=True,
            # break_when_any_done=False,
            # break_when_all_done=True,
        )


if __name__ == "__main__":
    main()
