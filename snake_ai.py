import os
from snake_env.snake_custom_env import SnakeCustomEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

CALLBACK_LOG_DIR = "\logs\\snake_ai"
CALLBACK_CHECKPOINT_DIR = "\models\\snake_ai"
CALLBACK_CHECK_FREQ = 10000
CALLBACK_ON_TRAINING_MODEL = "snake_ai_training_model"
CALLBACK_ON_TRAINING_END = "snake_ai_training_end_"

VERBOSE = 1

ALGORITHM_BEST_MODEL_NAME = "snake_ai_best_model.zip"
ALGORITHM_POLICY = "MlpPolicy"
ALGORITHM_DEVICE = "cpu"
ALGORITHM_TOTAL_TIMESTEPS = 3000000

ALGORITHM_RENDER_MODEL = True
ALGORITHM_RENDER_EPISODES = 100
ALGORITHM_PREDICT_DETERMINISTIC = False
ALGORITHM_RENDER_MODE = "human"

ALGORITHM_NEW_MODEL = True
ALGORITHM_LOAD_MODEL = False

MESSAGE_RENDERING_MODEL = "RENDERING MODEL"
MESSAGE_LOADING_MODEL = "LOADING MODEL"
MESSAGE_TRAINING_NEW_MODEL = "NEW MODEL"


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=VERBOSE):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, "{}".format(self.n_calls))
            self.model.save(model_path)

        return True

    def _on_training_end(self):
        model_path = os.path.join(
            self.save_path, f"{CALLBACK_ON_TRAINING_END}_{ALGORITHM_TOTAL_TIMESTEPS}"
        )
        self.model.save(model_path)

        return True

    def _on_training_end(self):
        model_path = os.path.join(
            self.save_path, f"{CALLBACK_ON_TRAINING_END}_{ALGORITHM_TOTAL_TIMESTEPS}"
        )
        self.model.save(model_path)


env = SnakeCustomEnv()
env.reset()
callback = TrainAndLoggingCallback(
    check_freq=CALLBACK_CHECK_FREQ, save_path=CALLBACK_CHECKPOINT_DIR
)

if ALGORITHM_RENDER_MODEL is True:
    print(MESSAGE_RENDERING_MODEL)
    model = PPO.load(ALGORITHM_BEST_MODEL_NAME, env=env)
    vec_env = model.get_env()

    for ep in range(ALGORITHM_RENDER_EPISODES):
        obs = vec_env.reset()
        dones = False
        while not dones:
            action, _ = model.predict(
                obs, deterministic=ALGORITHM_PREDICT_DETERMINISTIC
            )
            obs, rewards, dones, info = vec_env.step(action)

    env.close()

elif ALGORITHM_LOAD_MODEL is True:
    print(MESSAGE_LOADING_MODEL)
    model = PPO.load(ALGORITHM_BEST_MODEL_NAME, env=env, device=ALGORITHM_DEVICE)
    model.learn(total_timesteps=ALGORITHM_TOTAL_TIMESTEPS, callback=callback)

elif ALGORITHM_NEW_MODEL is True:
    print(MESSAGE_TRAINING_NEW_MODEL)
    model = PPO(
        ALGORITHM_POLICY,
        env,
        tensorboard_log=CALLBACK_LOG_DIR,
        verbose=VERBOSE,
        device=ALGORITHM_DEVICE,
    )
    model.learn(total_timesteps=ALGORITHM_TOTAL_TIMESTEPS, callback=callback)
