import rl_zoo3
import rl_zoo3.train

# from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.registration import register
from rl_zoo3.train import train
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

import gym_env

# Register the environment
gym_env.register_env()

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

# Register the environment
register(
    id="TinyPhysicsEnv-v0",
    entry_point="tinyphysics:TinyPhysicsEnv",
)


if __name__ == "__main__":
    train()
    # check_env(TinyPhysicsEnv())
