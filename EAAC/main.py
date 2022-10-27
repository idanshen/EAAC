from EAAC.wrappers import make
from omegaconf import DictConfig
from stable_baselines3 import SAC, EAAC
import gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import os


@hydra.main(config_path="config", config_name="test.yaml")
def main(cfg: DictConfig):
    env = make(**cfg.env)
    eval_env = make(**cfg.env)
    # model = SAC(policy=cfg.SAC.policy,
    #             env=env,
    #             learning_starts=cfg.SAC.learning_starts,
    #             target_entropy=cfg.SAC.target_entropy,
    #             train_freq=cfg.SAC.train_freq,
    #             gradient_steps=cfg.SAC.gradient_steps,
    #             gamma=cfg.SAC.gamma,
    #             verbose=1,
    #             tensorboard_log=os.getcwd(),
    #             )
    model = EAAC(policy=cfg.EAAC.policy,
                 env=env,
                 eval_env=eval_env,
                 learning_starts=cfg.EAAC.learning_starts,
                 target_entropy=cfg.EAAC.target_entropy,
                 trajectory_length=cfg.env.episode_length,
                 gradient_to_steps_ratio=cfg.EAAC.gradient_to_steps_ratio,
                 gamma=cfg.EAAC.gamma,
                 verbose=1,
                 tensorboard_log=os.getcwd(),
                 )
    model.learn(total_timesteps=cfg.SAC.n_timesteps)
    env.close()


if __name__ == "__main__":
    main()