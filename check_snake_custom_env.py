from stable_baselines3.common.env_checker import check_env
from snake_game_custom_env import SnakeCustomEnv

env = SnakeCustomEnv()
check_env(env)
