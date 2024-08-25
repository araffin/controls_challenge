from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.envs import registration

from controllers.pid import Controller as PIDController
from minimal import (
    ACC_G,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    # FUTURE_PLAN_STEPS,
    LAT_ACCEL_COST_MULTIPLIER,
    MAX_ACC_DELTA,
    MAX_ERROR_SUM,
    # MAX_JERK,
    MAX_LATACCEL,
    STEER_RANGE,
    V_MAX,
    FuturePlan,
    State,
    TinyPhysicsModel,
    get_data,
    get_state_target_futureplan,
)

CURRENT_DIR = Path(__file__).resolve().parent

OBS_FUTURE_PLAN_STEPS = 5  # FUTURE_PLAN_STEPS
EXP_REWARD_TEMP = 0.1
INVERSE_ERROR_EPS = 0.1
HYBRID_ERROR_THRESHOLD = 1.0


class RewardType(Enum):
    EXP_ERROR = "exp_error"
    EXP_RELATIVE = "exp_relative"
    L2_ERROR = "l2_error"
    L2_RELATIVE = "l2_relative"
    INVERSE_ERROR = "inverse_error"
    INVERSE_RELATIVE = "inverse_relative"
    HYBRID = "hybrid" # inverse error for small errors, l2 error for large errors


class LatAccelEnv(gym.Env):
    data: pd.DataFrame

    def __init__(
        self,
        max_range: float = 1.0,
        debug: bool = False,
        max_traj: int = 50,
        pid_coef: float = 0.0,
        reward_type: str = RewardType.EXP_ERROR.value,
        reward_discount: float = 1.0,
    ):
        super().__init__()

        data_path = CURRENT_DIR / "data"
        model_path = CURRENT_DIR / "models/tinyphysics.onnx"

        # Load the model
        self.model = TinyPhysicsModel(str(model_path), debug)
        self.debug = debug
        self.datasets = sorted(list(data_path.glob("*.csv")))[:max_traj]

        # Define action space and observation space
        self.max_range = max_range
        self.pid_coef = pid_coef
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # TODO: maybe include history?
        n_state = 3
        n_target = 1
        n_pid = 5
        # n_obs = n_state + n_target + n_pid
        n_obs = n_state + n_target + OBS_FUTURE_PLAN_STEPS * n_state + n_pid
        # n_obs = n_pid

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.state_history: list[State] = []
        self.action_history: list[float] = []
        self.current_lataccel_history: list[float] = []
        self.target_lataccel_history: list[float] = []
        self.step_idx = 0
        self.controller = PIDController()

        self.reward_type = RewardType(reward_type)
        self.next_reward_discount = reward_discount

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
        kp, ki, kd = 0.3, 0.05, -0.1
        pid_action = kp * current_error + ki * error_integral + kd * error_diff

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
        # return np.concatenate([state_obs, target_obs, pid_obs]).astype(np.float32)

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        # Choose a random dataset
        data_path = np.random.choice(self.datasets)  # type: ignore
        self.data = get_data(data_path)
        self.step_idx = CONTEXT_LENGTH

        self.state_history = []
        self.action_history = self.data["steer_command"].values[:CONTEXT_LENGTH].tolist()
        self.target_lataccel_history = []
        self.current_lataccel_history = []
        self.last_error = 0.0
        self.error_integral = 0.0
        self.error_diff = 0.0

        for i in range(CONTEXT_LENGTH):
            state, target, futureplan = get_state_target_futureplan(self.data, i)
            self.state_history.append(state)
            # Following perfectly the trajectory at the beginning
            self.target_lataccel_history.append(target)
            self.current_lataccel_history.append(target)

        # Warmup the trajectory using actions from the dataset
        assert CONTROL_START_IDX > CONTEXT_LENGTH
        for _ in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state, target, future_plan = get_state_target_futureplan(self.data, self.step_idx)
            self.state_history.append(state)
            self.target_lataccel_history.append(target)
            action = self.data["steer_command"].values[self.step_idx]
            self.action_history.append(action)
            current_lataccel = get_state_target_futureplan(self.data, self.step_idx)[1]
            self.current_lataccel_history.append(current_lataccel)
            current_error = target - current_lataccel
            self.error_diff = current_error - self.last_error
            self.last_error = current_error
            self.error_integral += self.last_error
            self.step_idx += 1

        self.error_integral = np.clip(self.error_integral, -MAX_ERROR_SUM, MAX_ERROR_SUM)

        obs = self.get_observation(
            state,
            current_lataccel,
            target,
            future_plan,
            self.last_error,
            self.error_integral,
        )
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Clip the action to valid range
        action = np.clip(action.item() * self.max_range, STEER_RANGE[0], STEER_RANGE[1])

        kp, ki, kd = 0.3, 0.05, -0.1
        pid_action = kp * self.last_error + ki * self.error_integral + kd * self.error_diff
        pid_action = np.clip(pid_action, STEER_RANGE[0], STEER_RANGE[1])

        # Blend PID and RL
        h = self.pid_coef
        action = h * pid_action + (1.0 - h) * action

        self.action_history.append(action)  # type: ignore[arg-type]

        # Simulate the model step
        pred = self.model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:],
        )
        pred = np.clip(
            pred,
            self.current_lataccel_history[-1] - MAX_ACC_DELTA,
            self.current_lataccel_history[-1] + MAX_ACC_DELTA,
        )

        current_lataccel = pred
        # if self.step_idx >= CONTROL_START_IDX:
        #     current_lataccel = pred
        # else:
        #     current_lataccel = get_state_target_futureplan(self.data, self.step_idx)[1]
        self.current_lataccel_history.append(current_lataccel)

        # jerk_penalty = 0.0
        # if self.step_idx > CONTROL_START_IDX:
        #     # TODO: check to penalize action jerk instead
        #     jerk_penalty = (self.current_lataccel_history[-1] - self.current_lataccel_history[-2]) ** 2

        last_target = self.target_lataccel_history[-1]
        # Calculate reward
        tracking_error = (last_target - current_lataccel) ** 2
        if self.reward_type == RewardType.EXP_ERROR:
            # Use bounded exponential reward
            reward = np.exp(-tracking_error / EXP_REWARD_TEMP)
        elif self.reward_type == RewardType.EXP_RELATIVE:
            # Use relative improvement as reward
            previous_reward = np.exp(-self.last_error**2 / EXP_REWARD_TEMP)
            current_reward = np.exp(-(tracking_error**2) / EXP_REWARD_TEMP)
            reward = self.next_reward_discount * current_reward - previous_reward
        elif self.reward_type == RewardType.L2_ERROR:
            # Use L2 error as reward
            # norm_factor = MAX_LATACCEL ** 2
            reward = -tracking_error
        elif self.reward_type == RewardType.L2_RELATIVE:
            # Use relative improvement as reward
            previous_penalty = self.last_error**2
            current_penalty = tracking_error
            reward = previous_penalty - self.next_reward_discount * current_penalty
        elif self.reward_type == RewardType.INVERSE_ERROR:
            # Use inverse error as reward
            reward = INVERSE_ERROR_EPS / (INVERSE_ERROR_EPS + tracking_error)
        elif self.reward_type == RewardType.INVERSE_RELATIVE:
            previous_reward = INVERSE_ERROR_EPS / (INVERSE_ERROR_EPS + self.last_error**2)
            current_reward = INVERSE_ERROR_EPS / (INVERSE_ERROR_EPS + tracking_error)
            reward = self.next_reward_discount * current_reward - previous_reward
        elif self.reward_type == RewardType.HYBRID:
            inverse_reward = INVERSE_ERROR_EPS / (INVERSE_ERROR_EPS + tracking_error)
            l2_penalty = tracking_error
            # Inverse reward dominates if tracking error is small
            reward = inverse_reward - l2_penalty
            # Other option:
            # above_threshold = tracking_error > HYBRID_THRESHOLD
            # reward = inverse_reward - above_threshold * l2_penalty
        else:
            raise NotImplementedError(f"Reward type {self.reward_type} not implemented")

        # tracking_penalty = -(tracking_error / MAX_LATACCEL * LAT_ACCEL_COST_MULTIPLIER)
        # jerk_penalty = -jerk_penalty / MAX_JERK

        # print(f"tracking_penalty: {tracking_penalty:>6.4}, jerk_penalty: {jerk_penalty:>6.4}")
        # reward = tracking_penalty + jerk_penalty
        # reward = tracking_penalty

        # Check if the episode is over
        terminated = False
        truncated = self.step_idx >= COST_END_IDX

        if truncated and self.debug:
            # Compute cost for full trajectory
            targets = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
            predictions = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

            lat_accel_cost = np.mean((targets - predictions) ** 2) * 100
            jerk_cost = np.mean((np.diff(predictions) / DEL_T) ** 2) * 100
            total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost

            print(f"lataccel_cost: {lat_accel_cost:>6.4}, jerk_cost: {jerk_cost:>6.4}, total_cost: {total_cost:>6.4}")

        self.step_idx += 1
        # New observation
        current_error = last_target - current_lataccel
        self.error_integral += current_error
        self.error_integral = np.clip(self.error_integral, -MAX_ERROR_SUM, MAX_ERROR_SUM)
        state, target, future_plan = get_state_target_futureplan(self.data, self.step_idx)
        obs = self.get_observation(
            state,
            current_lataccel,
            target,
            future_plan,
            self.last_error,
            self.error_integral,
        )
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.error_diff = current_error - self.last_error
        self.last_error = current_error

        return obs, reward, terminated, truncated, {}


def register_env() -> None:
    # Register the environment
    registration.register(
        id="LatAccel-v0",
        entry_point=LatAccelEnv,  # type: ignore
    )


if __name__ == "__main__":
    # Test the environment
    from stable_baselines3.common.env_checker import check_env

    env = LatAccelEnv()
    check_env(env)
    print(env.observation_space, env.action_space)
