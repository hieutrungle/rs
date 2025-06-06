import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


from collections import deque
from dataclasses import dataclass
import traceback
import importlib.resources
import sys
import numpy as np
import torch
import wandb
import pyrallis
from typing import Callable
import torch.multiprocessing

torch.multiprocessing.set_start_method("forkserver", force=True)
from torchrl.envs import check_env_specs
import torchrl
from torchrl.envs import ParallelEnv, EnvBase, SerialEnv
from tensordict import TensorDict, TensorDictBase
import time
import gc

torch.multiprocessing.set_start_method("forkserver", force=True)

import rs
from rs.utils import pytorch_utils, utils
from rs.envs import Classroom, ClassroomEval, TwoAgentDataCenter, Classroom4UE, Classroom2UE

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import (
    Transform,
    Compose,
    RewardSum,
    TransformedEnv,
    StepCounter,
    ObservationNorm,
    DoubleToFloat,
)
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

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
    seed: int = 10  # seed of the experiment
    eval_seed: int = 111  # seed of the evaluation
    save_interval: int = 100  # the interval to save the model
    start_step: int = 0  # the starting step of the experiment
    track_wandb: bool = True
    use_compile: bool = False  # whether to use torch.dynamo compiler
    image_dir: str = (
        "-1"  # the path to the image directory, if not provided, it will be set to source_dir
    )

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 3  # the number of parallel environments
    ep_len: int = 50  # the maximum length of an episode
    eval_ep_len: int = 50  # the maximum length of an episode

    # Sampling
    frames_per_batch: int = 200  # Number of team frames collected per training iteration
    n_iters: int = 500  # Number of sampling and training iterations

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

        # image_dir = sionna_config["image_dir"]
        image_dir = os.path.join(config.image_dir, scene_name)
        # image_dir = config.image_dir
        sionna_config["image_dir"] = image_dir

        seed = config.seed + idx

        if config.command.lower() == "train":
            sionna_config["rendering"] = False
        else:
            sionna_config["rendering"] = True

        if config.command.lower() == "train":
            if config.env_id.lower() == "classroom":
                env_cls = Classroom
            elif config.env_id.lower() == "classroom2ue":
                env_cls = Classroom2UE
            elif config.env_id.lower() == "classroom4ue":
                env_cls = Classroom4UE
            elif config.env_id.lower() == "data_center":
                env_cls = TwoAgentDataCenter
            else:
                raise ValueError(f"Unknown environment id: {config.env_id}")
            env = env_cls(
                sionna_config,
                seed=seed,
                device=config.device,
                num_runs_before_restart=20,
            )
        elif config.command.lower() == "eval":
            if config.env_id.lower() == "classroom":
                env_cls = ClassroomEval
            elif config.env_id.lower() == "classroom2ue":
                env_cls = Classroom2UE
            elif config.env_id.lower() == "classroom4ue":
                env_cls = Classroom4UE
            elif config.env_id.lower() == "data_center":
                env_cls = TwoAgentDataCenter
            else:
                raise ValueError(f"Unknown environment id: {config.env_id}")
            env = env_cls(
                sionna_config,
                seed=config.eval_seed + idx,
                device=config.device,
                num_runs_before_restart=20,
                eval_mode=True,
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

    if loc is None and scale is None:
        envs.transform[0].init_stats(num_iter=config.ep_len * 3, reduce_dim=(0, 1, 2), cat_dim=1)

    try:
        if envs.is_closed:
            envs.start()
        n_agents = list(envs.n_agents)[0]
        shared_parameters_policy = False
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                # n_obs_per_agent
                n_agent_inputs=ob_spec["agents", "observation"].shape[-1],
                n_agent_outputs=2 * ac_spec.shape[-1],  # 2 * n_actions_per_agents
                n_agents=n_agents,
                centralised=False,
                share_params=shared_parameters_policy,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.ReLU6,
            ),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=envs.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[envs.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": envs.full_action_spec_unbatched[envs.action_key].space.low,
                "high": envs.full_action_spec_unbatched[envs.action_key].space.high,
            },
            return_log_prob=True,
        )  # we'll need the log-prob for the PPO loss

        share_parameters_critic = False
        mappo = True

        critic_net = MultiAgentMLP(
            n_agent_inputs=ob_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=n_agents,
            centralised=mappo,
            share_params=share_parameters_critic,
            device=config.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        critic = TensorDictModule(
            module=critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )

        collector = SyncDataCollector(
            envs,
            policy,
            device=config.device,
            storing_device=config.device,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.total_frames,
        )

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(config.frames_per_batch, device=config.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=config.minibatch_size,
        )

        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            loss_critic_type="l2",
            normalize_advantage=True,
            normalize_advantage_exclude_dims=(1,),
            clip_epsilon=config.clip_epsilon,
            entropy_bonus=config.entropy_eps > 0,
            entropy_coef=config.entropy_eps,
            # Important to avoid normalizing across the agent dimension
            # normalize_advantage=False,
        )
        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=envs.reward_key,
            action=envs.action_key,
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        # GAE
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda
        )

        # optimizer
        optim = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

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
    critic: TensorDictModule,
    loss_module: ClipPPOLoss,
    optim: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
):

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
    GAE = loss_module.value_estimator
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

        with torch.no_grad():
            # Compute GAE and add it to the data
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        for _ in range(config.num_epochs):
            for _ in range(config.frames_per_batch // config.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.max_grad_norm)
                optim.step()
                optim.zero_grad()

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
            "loss_objective": loss_vals["loss_objective"].item(),
            "loss_critic": loss_vals["loss_critic"].item(),
            "loss_entropy": loss_vals["loss_entropy"].item(),
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
        collector = SyncDataCollector(
            envs,
            policy,
            device=config.device,
            storing_device=config.device,
            frames_per_batch=config.ep_len * 20 * config.num_envs,
            total_frames=config.ep_len * 20 * config.num_envs,
        )
        for idx, tensordict_data in enumerate(collector):
            rollouts = tensordict_data

    # save the rollout
    image_dir = config.image_dir
    # get last name of the image directory
    name = image_dir.split("/")[-1]
    rollout_path = os.path.join(config.checkpoint_dir, f"env_rollout_{name}.pt")
    torch.save(rollouts, rollout_path)
    print(f"Rollout saved to {rollout_path}")


if __name__ == "__main__":
    main()
