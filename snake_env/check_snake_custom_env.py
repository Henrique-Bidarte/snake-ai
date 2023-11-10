from stable_baselines3.common.env_checker import check_env
from snake_game_custom_env import SnakeCustomEnv

env = SnakeCustomEnv()
check_env(env)

CHECK_EPISODES = 50

for episode in range(CHECK_EPISODES):
    terminated = False
    truncated = False
    obs = env.reset()
    while not terminated or not truncated:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        print("reward", reward)
