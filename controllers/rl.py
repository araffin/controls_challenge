import pickle
from collections import namedtuple

import numpy as np
import sbx

from . import BaseController

MAX_LATACCEL = 5
# For normalization
V_MAX = 50.0
OBS_FUTURE_PLAN_STEPS = 5
ACC_G = 9.81
MAX_ERROR_SUM = 10.0

State = namedtuple("State", ["roll_lataccel", "v_ego", "a_ego"])
FuturePlan = namedtuple("FuturePlan", ["lataccel", "roll_lataccel", "v_ego", "a_ego"])


class Controller(BaseController):
    def __init__(
        self,
    ):
        model_path = "./logs/ppo/LatAccel-v0_23/"
        self.rl_model = sbx.PPO.load(f"{model_path}/best_model.zip")
        self.action_scale = 0.5
        # self.vec_norm = VecNormalize.load(f"{model_path}/LatAccel-v0/vecnormalize.pkl")
        with open(f"{model_path}/LatAccel-v0/vecnormalize.pkl", "rb") as file_handler:
            self.vec_norm = pickle.load(file_handler)

        self.last_error = 0.0
        self.error_integral = 0.0
        self.pid_coef = 0.5

    @staticmethod
    def pid_action(error: float, error_integral: float, last_error: float) -> float:
        kp, ki, kd = 0.3, 0.05, -0.1
        pid_action = kp * error + ki * error_integral + kd * (error - last_error)
        return np.clip(pid_action, -2.0, 2.0)

    @staticmethod
    def get_observation(
        state: State,
        current_lataccel: float,
        target: float,
        future_plan: FuturePlan,
        last_error: float,
        error_integral: float,
    ) -> np.ndarray:
        future_plan_obs = np.zeros(3 * OBS_FUTURE_PLAN_STEPS)
        state_obs = np.array([state.roll_lataccel / ACC_G, state.v_ego / V_MAX, state.a_ego])
        target_obs = np.array([target])
        # future plan can be less than FUTURE_PLAN_STEPS at the end of the trajectory
        # in that case, pad with zeros
        n_future_steps = min(len(future_plan.lataccel), OBS_FUTURE_PLAN_STEPS)
        future_plan_obs[:n_future_steps] = np.array(future_plan.lataccel)[:n_future_steps] / MAX_LATACCEL
        future_plan_obs[OBS_FUTURE_PLAN_STEPS : OBS_FUTURE_PLAN_STEPS + n_future_steps] = (
            np.array(future_plan.v_ego)[:n_future_steps] / V_MAX
        )
        future_plan_obs[2 * OBS_FUTURE_PLAN_STEPS : 2 * OBS_FUTURE_PLAN_STEPS + n_future_steps] = np.array(future_plan.a_ego)[
            :n_future_steps
        ]
        # Preprocess and give error as input too
        current_error = target - current_lataccel
        error_diff = current_error - last_error
        pid_action = Controller.pid_action(current_error, error_integral, last_error)

        pid_obs = np.array(
            [
                current_lataccel / MAX_LATACCEL,
                current_error,
                error_diff,
                error_integral,
                np.clip(pid_action, -2.0, 2.0),
            ]
        )
        # return pid_obs.astype(np.float32).flatten()
        # Concatenate all observations
        return np.concatenate([state_obs, target_obs, future_plan_obs, pid_obs]).astype(np.float32)

    def update(self, target_lataccel: float, current_lataccel: float, state: State, future_plan: FuturePlan) -> float:
        current_error = target_lataccel - current_lataccel
        self.error_integral += current_error
        self.error_integral = np.clip(self.error_integral, -MAX_ERROR_SUM, MAX_ERROR_SUM)

        obs = self.get_observation(state, current_lataccel, target_lataccel, future_plan, self.last_error, self.error_integral)
        norm_obs = self.vec_norm.normalize_obs(obs)
        action, _ = self.rl_model.predict(norm_obs, deterministic=True)
        action *= self.action_scale

        pid_action = Controller.pid_action(current_error, self.error_integral, self.last_error)

        # Blend PID and RL
        h = self.pid_coef
        action = h * pid_action + (1.0 - h) * float(action)

        self.last_error = current_error

        return action
