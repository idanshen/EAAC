from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.buffers import ReplayBuffer, TrajectoryReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3.eaac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit, ReplayBufferSamples
from stable_baselines3.common.callbacks import BaseCallback

EAACSelf = TypeVar("EAACSelf", bound="EAAC")


class EAAC(OffPolicyAlgorithm):
    """
    Entropy Augmented Actor-Critic

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        trajectory_length: int=1000,
        gradient_to_steps_ratio: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq=(10, "episode"),
            gradient_steps=2*gradient_to_steps_ratio*trajectory_length,
            action_noise=action_noise,
            replay_buffer_class=TrajectoryReplayBuffer,
            replay_buffer_kwargs={'trajectory_length': trajectory_length, 'save_log_prob': True},
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )
        self.entropy_update = 'new_method'
        self.trajectory_length = trajectory_length
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        self.R_policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.R_policy = self.R_policy.to(self.device)

        self._create_aliases()
        # Running mean and running var
        self.EA_batch_norm_stats = get_parameters_by_name(self.EA_critic, ["running_"])
        self.EA_batch_norm_stats_target = get_parameters_by_name(self.EA_critic_target, ["running_"])
        self.R_batch_norm_stats = get_parameters_by_name(self.R_critic, ["running_"])
        self.R_batch_norm_stats_target = get_parameters_by_name(self.R_critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # We save the log entropy coefficient for numerical stability but update the entropy coefficient itself
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(False)
            self.ent_coef_optimizer = True # th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.EA_policy = self.policy
        self.EA_actor = self.EA_policy.actor
        self.EA_critic = self.EA_policy.critic
        self.EA_critic_target = self.EA_policy.critic_target
        self.R_actor = self.R_policy.actor
        self.R_critic = self.R_policy.critic
        self.R_critic_target = self.R_policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.EA_actor.optimizer, self.EA_critic.optimizer, self.R_actor.optimizer, self.R_critic.optimizer]
        # if self.ent_coef_optimizer is not None:
        #     optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        EA_actor_losses, EA_critic_losses = [], []
        R_actor_losses, R_critic_losses = [], []
        # Policies optimization
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef)
            else:
                ent_coef = self.ent_coef_tensor

            critic_loss, actor_loss = self.update_step(replay_data=replay_data,
                                                       ent_coef=ent_coef,
                                                       actor=self.EA_actor,
                                                       critic=self.EA_critic,
                                                       critic_target=self.EA_critic_target)
            EA_critic_losses.append(critic_loss.item())
            EA_actor_losses.append(actor_loss.item())

            critic_loss, actor_loss = self.update_step(replay_data=replay_data,
                                                       ent_coef=th.zeros_like(ent_coef),
                                                       actor=self.R_actor,
                                                       critic=self.R_critic,
                                                       critic_target=self.R_critic_target)
            R_critic_losses.append(critic_loss.item())
            R_actor_losses.append(actor_loss.item())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.EA_critic.parameters(), self.EA_critic_target.parameters(), self.tau)
                polyak_update(self.R_critic.parameters(), self.R_critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.EA_batch_norm_stats, self.EA_batch_norm_stats_target, 1.0)
                polyak_update(self.R_batch_norm_stats, self.R_batch_norm_stats_target, 1.0)


        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/EA_actor_loss", np.mean(EA_actor_losses))
        self.logger.record("train/EA_critic_loss", np.mean(EA_critic_losses))
        self.logger.record("train/R_actor_loss", np.mean(R_actor_losses))
        self.logger.record("train/R_critic_loss", np.mean(R_critic_losses))


    def learn(
        self: EAACSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> EAACSelf:

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.policy = self.EA_policy
            rollout, J_EA = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            self.policy = self.R_policy
            _, J_R = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            self.policy = self.EA_policy
            J_R_est = J_R
            J_EA_est = J_EA

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    ent_coefs, ent_coef_losses = self.update_entropy(batch_size=self.batch_size,
                                                                     gradient_steps=gradient_steps,
                                                                     J_EA_est=J_EA_est,
                                                                     J_R_est=J_R_est)
                    self.logger.record("train/ent_coef", np.mean(ent_coefs))
                    if len(ent_coef_losses) > 0:
                        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
                    self.logger.record("train/J_EA_est", J_EA_est)
                    self.logger.record("train/J_R_est", J_R_est)

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        save_results: bool = True,
    ) -> Tuple[RolloutReturn, np.array]:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        : param save_results: Either to save episode results to buffer and results info or just return reward
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.policy.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        self.env.reset()
        cum_reward_list = []

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions, log_prob = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            for i, info in enumerate(infos): info['log_prob'] = log_prob[i]

            num_collected_steps += 1
            if save_results:
                self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            if save_results:
                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            else:
                self._last_obs = new_obs

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    if save_results:
                        self._episode_num += 1
                        # Log training infos
                        if log_interval is not None and self._episode_num % log_interval == 0:
                            self._dump_logs()

                    # Currently support only single env
                    cum_reward = infos[idx]['episode']['r']
                    cum_reward_list.append(cum_reward)
                    save_results = False

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training), np.mean(cum_reward_list)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["EA_actor", "EA_critic", "EA_critic_target", "R_actor", "R_critic", "R_critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["EA_policy", "EA_actor.optimizer", "EA_critic.optimizer", "R_policy", "R_actor.optimizer", "R_critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def update_step(self, replay_data: ReplayBufferSamples, actor: nn.Module, critic: nn.Module, critic_target: nn.Module, ent_coef: th.Tensor):
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert not th.isnan(critic_loss)
        # Optimize the critic
        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = th.cat(critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        assert not th.isnan(actor_loss)
        # Optimize the actor
        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

        return critic_loss, actor_loss

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                if self.policy == self.EA_policy:
                    self.ep_info_buffer.extend([maybe_ep_info])
                elif self.policy == self.R_policy:
                    pass
                else:
                    raise ValueError
            if maybe_is_success is not None and dones[idx]:
                if self.policy == self.EA_policy:
                    self.ep_success_buffer.extend(maybe_is_success)
                elif self.policy == self.R_policy:
                    pass
                else:
                    raise ValueError

    def update_entropy(self, gradient_steps: int, batch_size: int, J_EA_est: float = None,  J_R_est: float = None, ) -> Tuple[List[th.Tensor], List[th.Tensor]]:
        # Entropy coefficent optimization
        if self.entropy_update == 'original_SAC':
            ent_coef_losses, ent_coefs = [], []
            for gradient_step in range(gradient_steps):
                # Sample replay buffer
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

                # We need to sample because `log_std` may have changed between two gradient steps
                if self.use_sde:
                    self.EA_actor.reset_noise()

                # Action by the current actor for the sampled state
                actions_pi, log_prob = self.EA_actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None:
                    # Important: detach the variable from the graph
                    # so we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor

                ent_coefs.append(ent_coef.item())

                # Optimize entropy coefficient, also called
                # entropy temperature or alpha in the paper
                if ent_coef_loss is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                return ent_coefs, ent_coef_losses
        elif self.entropy_update == 'new_method':
            ent_coef_losses, ent_coefs = [], []
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_grad = (J_EA_est - J_R_est)
                ent_coef_losses.append(ent_coef_grad)
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            beta = 0.003
            self.log_ent_coef = th.clip((self.log_ent_coef) + beta * ent_coef_grad, -5, 5)

            return ent_coefs, ent_coef_losses
        else:
            raise ValueError("self.entropy_update must be either original_SAC or new_method")
