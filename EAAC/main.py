import matplotlib.pyplot as plt
from stable_baselines3 import SAC, EAAC
import numpy as np
from EAAC.wrappers import make
import hydra
from omegaconf import DictConfig
import gym
from torch.utils.tensorboard import SummaryWriter
import os


@hydra.main(config_path="config", config_name="test.yaml")
def main(cfg: DictConfig):
    env = make(**cfg.env)
    model = SAC(policy=cfg.SAC.policy,
                env=env,
                learning_starts=cfg.SAC.learning_starts,
                target_entropy=cfg.SAC.target_entropy,
                train_freq=cfg.SAC.train_freq,
                gradient_steps=cfg.SAC.gradient_steps,
                verbose=1,
                tensorboard_log=os.getcwd(),
                )
    model = EAAC(policy=cfg.EAAC.policy,
                 env=env,
                 learning_starts=cfg.EAAC.learning_starts,
                 target_entropy=cfg.EAAC.target_entropy,
                 train_freq=cfg.EAAC.train_freq,
                 gradient_steps=cfg.EAAC.gradient_steps,
                 verbose=1,
                 tensorboard_log=os.getcwd(),
                 )
    model.learn(total_timesteps=cfg.SAC.n_timesteps)
    env.close()

    # obs = env.reset()
    # rewards = []
    # for i in range(1001):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     rewards.append(reward)
    #     image = env.render(mode='rgb_array')
    #     plt.imshow(image)
    #     plt.draw()
    #     plt.pause(0.0001)
    #     if done:
    #         obs = env.reset()
    # print(np.mean(rewards))
    # plt.show()


if __name__ == "__main__":
    main()