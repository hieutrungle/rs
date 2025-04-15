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
from torch import multiprocessing

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

from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder

from tqdm import tqdm

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
    frames_per_batch: int = 500  # Number of team frames collected per sampling iteration
    n_iters: int = 200  # Number of sampling and training iterations

    # Replay buffer
    memory_size: int = 50_000  # The replay buffer of each group can store this many frames

    # Training
    num_epochs: int = 100  # Number of optimization steps per training iteration
    minibatch_size: int = 128  # Size of the mini-batches in each optimization step
    lr: float = 3e-4  # Learning rate
    max_grad_norm: float = 1.0  # Maximum norm for the gradients

    # DDPG
    gamma: float = 0.99  # Discount factor
    polyak_tau: float = 0.005  # Tau for the soft-update of the target network

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
    if config.verbose:
        utils.log_args(config)

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
    envs.transform[0].init_stats(num_iter=int(config.ep_len * 3), reduce_dim=(0, 1, 2), cat_dim=1)
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
            out_keys=[("agents", "param")],
            # out_keys=[("agents", "loc"), ("agents", "scale")],
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
            # Read ``("agents", "obs_action")``
            in_keys=[("agents", "obs_action")],
            # Write ``("agents", "state_action_value")``
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

        # scratch_dir = tempfile.TemporaryDirectory().name
        # scratch_dirs.append(scratch_dir)
        replay_buffer = ReplayBuffer(
            # We will store up to memory_size multi-agent transitions
            storage=LazyMemmapStorage(config.memory_size, scratch_dir=config.replay_buffer_dir),
            sampler=RandomSampler(),
            batch_size=config.minibatch_size,  # We will sample batches of this size
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

        target_updater = SoftUpdate(loss_module, tau=config.polyak_tau)
        optimizers = {
            "loss_actor": torch.optim.Adam(
                loss_module.actor_network_params.flatten_keys().values(), lr=config.lr
            ),
            "loss_value": torch.optim.Adam(
                loss_module.value_network_params.flatten_keys().values(), lr=config.lr
            ),
        }

        def process_batch(tensordict_data: TensorDictBase) -> TensorDictBase:
            """
            If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
            `"terminated"` and `"done"`.
            This is needed to present them with the same shape as the reward to the loss.
            """
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

        pbar = tqdm(
            total=config.n_iters,
            desc=", ".join([f"episode_reward_mean_{group} = 0" for group in ["agents"]]),
            dynamic_ncols=True,
        )
        episode_reward_mean_map = {group: [] for group in ["agents"]}

        # Training/collection iterations
        for iteration, batch in enumerate(collector):
            current_frames = batch.numel()
            process_batch(batch)  # Util to expand done keys if needed
            # Loop over groups
            # This just affects the leading dimensions in batch_size of the tensordict
            batch = batch.reshape(-1)
            replay_buffer.extend(batch)

            for _ in range(config.num_epochs):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                for loss_name in ["loss_actor", "loss_value"]:
                    loss = loss_vals[loss_name]
                    optimizer = optimizers[loss_name]

                    loss.backward()

                    # Optional
                    params = optimizer.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                # Soft-update the target network
                target_updater.step()

            # Exploration sigma anneal update
            exploration_policy[-1].step(current_frames)

            # Logging
            episode_reward_mean = (
                batch.get(("next", "agents", "episode_reward"))[
                    batch.get(("next", "agents", "done"))
                ]
                .mean()
                .item()
            )
            episode_reward_mean_map["agents"].append(episode_reward_mean)
            wandb.log(
                {
                    f"episode_reward_mean_agents": episode_reward_mean,
                    "loss_actor": loss_vals["loss_actor"].item(),
                    "loss_value": loss_vals["loss_value"].item(),
                },
                step=iteration,
            )
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "critic": critic.state_dict(),
                    "loss_module": loss_module.state_dict(),
                },
                os.path.join(config.checkpoint_dir, f"checkpoint_{iteration}.pt"),
            )

            pbar.set_description(
                ", ".join(
                    [
                        f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                        for group in ["agents"]
                    ]
                ),
                refresh=False,
            )
            pbar.update()

        fig, axs = plt.subplots(1, 1)
        for i, group in enumerate(["agents"]):
            axs.plot(episode_reward_mean_map[group], label=f"Episode reward mean {group}")
            axs.set_ylabel("Reward")
        axs.set_xlabel("Training iterations")
        plt.show()

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
    try:
        check_env_specs(env)
        # n_rollout_steps = 5
        # rollout = env.rollout(n_rollout_steps)
        # # rollout has batch_size of (num_vmas_envs, n_rollout_steps)

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
