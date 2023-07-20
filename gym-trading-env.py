# from stable_baselines3 import A2C
import gym
import gym_anytrading
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(DIR_PATH + "/data/TATASTEEL.csv", index_col="Date", parse_dates=True)

train_env = gym.make('stocks-v0',df=df, window_size=20, frame_bound=(20, 1000))

model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=DIR_PATH+"/logs")
model.batch_size = 128


for i in range(99):
    model.learn(total_timesteps=10000, tb_log_name="PPO_demo_1", reset_num_timesteps=False)
    model.save(DIR_PATH + f"/models/PPO/PPO_demo_{10*(i+1)}k_steps")