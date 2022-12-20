

from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from EAAC.wrappers import make
from omegaconf import DictConfig
from stable_baselines3 import SAC, EAAC, PPO
import hydra
import matplotlib.pyplot as plt
import numpy as np
import os
import gymnasium as gym


import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy


@hydra.main(config_path="config", config_name="test.yaml")
def main(cfg: DictConfig):
    env = gym.make("FetchReach-v2")
    # model = SAC(policy=cfg.SAC.policy,
    #             env=env,
    #             learning_starts=cfg.SAC.learning_starts,
    #             target_entropy=cfg.SAC.target_entropy,
    #             ent_coef=cfg.SAC.ent_coef,
    #             train_freq=cfg.SAC.train_freq,
    #             gradient_steps=cfg.SAC.gradient_steps,
    #             gamma=cfg.SAC.gamma,
    #             verbose=1,
    #             tensorboard_log=os.getcwd(),
    #             )
    # model = EAAC(policy=cfg.EAAC.policy,
    #              env=env,
    #              learning_starts=cfg.EAAC.learning_starts,
    #              target_entropy=cfg.EAAC.target_entropy,
    #              trajectory_length=cfg.env.episode_length,
    #              gradient_to_steps_ratio=cfg.EAAC.gradient_to_steps_ratio,
    #              gamma=cfg.EAAC.gamma,
    #              verbose=1,
    #              tensorboard_log=os.getcwd(),
    #              )
    # model.learn(total_timesteps=cfg.EAAC.n_timesteps)
    model = RecurrentPPO("MlpLstmPolicy",
                 env=env,
                 verbose=1,
                 tensorboard_log=os.getcwd(),
                 )
    model.learn(total_timesteps=5000)
    env.close()


if __name__ == "__main__":
    main()
