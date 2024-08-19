import numpy as np
from custom_envs.filter_wrappers import ActionFilterButter
from sbx import CrossQ

from . import BaseController


class Controller(BaseController):
    """
    A simple PID controller
    """

    def __init__(
        self,
    ):
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0
        self._action_filter = ActionFilterButter(
            sampling_rate=60,
            num_joints=1,
            lowcut=[0.0],
            highcut=[4.0],
        )
        self._action_filter.reset()
        self._action_filter.init_history(np.zeros(1))

        self.rl_model = CrossQ.load("./logs/TinyPhysicsEnv-v0_7/best_model.zip")
        self.action_scale = 1.0
        self.last_action = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan) -> float:
        # PID
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff

        # RL
        next_lateral_accel = np.zeros(3)
        n_next = min(len(future_plan.lataccel), 3)
        next_lateral_accel[:n_next] = target_lataccel
        next_lateral_accel[:n_next] -= future_plan.lataccel[:n_next]
        obs_pid = np.array(
            [
                error,
                error_diff,
                np.clip(self.error_integral, -5, 5),
                self.last_action,
            ]
        )
        rl_obs = np.array([*obs_pid, *next_lateral_accel]).astype(np.float32)
        rl_action = self.rl_model.predict(rl_obs, deterministic=True)[0]
        rl_action = self._action_filter.filter(rl_action)

        action = float(pid_action + self.action_scale * rl_action)
        action = np.clip(action, -2.0, 2.0)
        self.last_action = action
        return action
