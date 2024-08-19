import numpy as np
import sbx
from custom_envs.filter_wrappers import ActionFilterButter

from . import BaseController


class HistoryWrapper:
    """
    Stack past observations and actions to give an history to the agent.
    """

    def __init__(self, obs_size: int = 7, n_actions: int = 1, horizon: int = 2):
        self.horizon = horizon
        self.obs_history = np.zeros((horizon * obs_size,), np.float32)
        self.action_history = np.zeros((horizon * n_actions,), np.float32)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, obs: np.ndarray):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history()

    def step(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action
        return self._create_obs_from_history()


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

        # Silence user warnings
        # import warnings
        # warnings.filterwarnings("ignore")

        self.rl_model = sbx.CrossQ.load(
            "./logs/crossq/TinyPhysicsEnv-v0_13/best_model.zip",
            # custom_objects={
            #     "actor": None,
            #     "lr_schedule": None,
            #     "ent_coef_state": None,
            #     "policy": None,
            #     "qf": None,
            # },
        )
        self.action_scale = 0.5
        self.last_action = 0.0
        self.history_wrapper = HistoryWrapper()

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
        rl_obs = self.history_wrapper.step(rl_obs, np.array([self.last_action]))

        rl_action = self.rl_model.predict(rl_obs, deterministic=True)[0]
        rl_action = self._action_filter.filter(rl_action)

        action = float(pid_action + self.action_scale * rl_action)
        action = np.clip(action, -2.0, 2.0)
        self.last_action = action
        return action
