from collections import namedtuple

import numpy as np
import sbx

from . import BaseController

FPS = 10
MAX_LATACCEL = 5
# For normalization
V_MAX = 50.0
MAX_JERK = 1  # TODO: tune this

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs


State = namedtuple("State", ["roll_lataccel", "v_ego", "a_ego"])
FuturePlan = namedtuple("FuturePlan", ["lataccel", "roll_lataccel", "v_ego", "a_ego"])


class Controller(BaseController):
    def __init__(
        self,
    ):
        self.rl_model = sbx.CrossQ.load(
            "./logs/crossq/LatAccel-v0_13/best_model.zip",
        )
        self.action_scale = 2.0

    @staticmethod
    def get_observation(state: State, current_lataccel: float, target: float, future_plan: FuturePlan) -> np.ndarray:
        future_plan_obs = np.zeros(3 * FUTURE_PLAN_STEPS)
        state_obs = np.array([state.roll_lataccel / MAX_LATACCEL, state.v_ego / V_MAX, state.a_ego])
        target_obs = np.array([target])
        # future plan can be less than FUTURE_PLAN_STEPS at the end of the trajectory
        # in that case, pad with zeros
        n_future_steps = len(future_plan.lataccel)
        future_plan_obs[:n_future_steps] = np.array(future_plan.lataccel) / MAX_LATACCEL
        future_plan_obs[FUTURE_PLAN_STEPS : FUTURE_PLAN_STEPS + n_future_steps] = np.array(future_plan.v_ego) / V_MAX
        future_plan_obs[2 * FUTURE_PLAN_STEPS : 2 * FUTURE_PLAN_STEPS + n_future_steps] = np.array(future_plan.a_ego)
        # Preprocess and give error as input too
        pid_obs = np.array([current_lataccel, (target - current_lataccel)]) / MAX_LATACCEL
        # Concatenate all observations
        return np.concatenate([state_obs, target_obs, future_plan_obs, pid_obs]).astype(np.float32)

    def update(self, target_lataccel: float, current_lataccel: float, state: State, future_plan: FuturePlan) -> float:
        obs = self.get_observation(state, current_lataccel, target_lataccel, future_plan)
        action = self.rl_model.predict(obs, deterministic=True)
        return action
