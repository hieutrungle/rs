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

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import (
    Compose,
    RewardSum,
    TransformedEnv,
    StepCounter,
    ObservationNorm,
    DoubleToFloat,
)
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
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

    use_compile: bool = False  # whether to use torch.dynamo compiler

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 3  # the number of parallel environments
    ep_len: int = 75  # the maximum length of an episode
    eval_ep_len: int = 75  # the maximum length of an episode

    # Sampling
    frames_per_batch: int = 500  # Number of team frames collected per training iteration
    n_iters: int = 200  # Number of sampling and training iterations

    # Training
    num_epochs: int = 40  # Number of optimization steps per training iteration
    minibatch_size: int = 100  # Size of the mini-batches in each optimization step
    lr: float = 3e-4  # Learning rate
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

    envs = transform_envs(envs, config)
    envs.transform[0].init_stats(num_iter=config.ep_len * 3, reduce_dim=(0, 1, 2), cat_dim=1)
    # batch_size rollout:
    # (num_envs, env_batch, n_rollout_steps) = (num_envs, 1, n_rollout_steps)

    try:
        wandb_init(config)

        if envs.is_closed:
            envs.start()
        n_agents = list(envs.n_agents)[0]
        shared_parameters_policy = False
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                # n_obs_per_agent
                n_agent_inputs=ob_spec["agents", "observation"].shape[-1],
                # 2 * n_actions_per_agents
                n_agent_outputs=2 * ac_spec.shape[-1],  # 2 * n_actions_per_agents
                n_agents=n_agents,
                #  the policies are decentralised (ie each agent will act from its observation)
                centralised=False,
                share_params=shared_parameters_policy,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            #  this will just separate the last dimension into two outputs: a loc and a non-negative scale
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
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_eps,
            # Important to avoid normalizing across the agent dimension
            normalize_advantage=False,
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
        GAE = loss_module.value_estimator

        # optimizer
        optim = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

        if config.load_model != "-1":
            # Load the model
            checkpoint = torch.load(config.load_model)
            policy.load_state_dict(checkpoint["policy"])
            critic.load_state_dict(checkpoint["critic"])
            loss_module.load_state_dict(checkpoint["loss_module"])
            optim.load_state_dict(checkpoint["optimizer"])

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
            episode_reward_mean_list.append(episode_reward_mean)
            logs = {
                "train/episode_reward_mean": episode_reward_mean,
                "train/loss_objective": loss_vals["loss_objective"].item(),
                "train/loss_critic": loss_vals["loss_critic"].item(),
                "train/loss_entropy": loss_vals["loss_entropy"].item(),
            }
            wandb.log({**logs}, step=idx)
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

    except Exception as e:
        print("Environment specs are not correct")
        print(e)
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if not envs.is_closed:
            envs.close()

    # print("Total step:", total_step)
    # print(f"Time taken for rollout: {end_time - start_time:.4f} s")
    # print(f"Average time per step: {(end_time - start_time) / total_step:.4f} s")
    # print(f"Average time per step in ms: {(end_time - start_time) / total_step * 1000:.4f} ms")


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


@pyrallis.wrap()
def test_env(config: TrainConfig):
    torch.compiler.reset()

    # Force torchrl to use forkserver for multiprocessing
    torch.multiprocessing.set_start_method("forkserver", force=True)

    # Load Sionna configuration
    # sionna_config = utils.load_config(config.sionna_config_file)
    pytorch_utils.init_seed(config.seed)
    env = make_env(config, 0)()
    env = TransformedEnv(
        env, RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
    )
    env.append_transform(StepCounter(max_steps=config.ep_len))
    # print(f"action_spec: {env.action_spec}\n")
    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)
    # print("done_spec:", env.full_done_spec)
    # print("action_keys:", env.action_keys)
    # print("reward_keys:", env.reward_keys)
    # print("done_keys:", env.done_keys)
    # print("observation_spec:", env.observation_keys)
    try:
        check_env_specs(env)
        # n_rollout_steps = 5
        # rollout = env.rollout(n_rollout_steps)
        # # rollout has batch_size of (num_vmas_envs, n_rollout_steps)

        # print("rollout of three steps:", rollout)
        # print("Shape of the rollout TensorDict:", rollout.batch_size)
        shared_parameters_policy = True
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                # n_obs_per_agent
                n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
                # 2 * n_actions_per_agents
                n_agent_outputs=2 * env.action_spec.shape[-1],  # 2 * n_actions_per_agents
                n_agents=env.n_agents,
                #  the policies are decentralised (ie each agent will act from its observation)
                centralised=False,
                share_params=shared_parameters_policy,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            #  this will just separate the last dimension into two outputs: a loc and a non-negative scale
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.full_action_spec_unbatched[env.action_key].space.low,
                "high": env.full_action_spec_unbatched[env.action_key].space.high,
            },
            return_log_prob=True,
        )  # we'll need the log-prob for the PPO loss

        share_parameters_critic = True
        mappo = True

        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=env.n_agents,
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
            env,
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
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_eps,
            # Important to avoid normalizing across the agent dimension
            normalize_advantage=False,
        )
        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=env.reward_key,
            action=env.action_key,
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        # GAE
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda
        )
        GAE = loss_module.value_estimator

        optim = torch.optim.Adam(loss_module.parameters(), lr=config.lr)
        pbar = tqdm(total=config.n_iters, desc="episode_reward_mean = 0.0")

        episode_reward_mean_list = []
        for idx, tensordict_data in enumerate(collector):

            # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
            tensordict_data.set(
                ("next", "agents", "done"),
                tensordict_data.get(("next", "done"))
                .unsqueeze(-1)
                .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
            )
            tensordict_data.set(
                ("next", "agents", "terminated"),
                tensordict_data.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
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
            episode_reward_mean_list.append(episode_reward_mean)
            pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
            pbar.update()

    except Exception as e:
        print("Environment specs are not correct")
        print(e)
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
