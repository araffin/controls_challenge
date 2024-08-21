from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.envs import registration

from controllers.pid import Controller as PIDController
from minimal import (
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    FUTURE_PLAN_STEPS,
    LAT_ACCEL_COST_MULTIPLIER,
    MAX_ACC_DELTA,
    MAX_ERROR_SUM,
    MAX_JERK,
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


class LatAccelEnv(gym.Env):
    data: pd.DataFrame

    def __init__(self, max_range: float = 1.0, debug: bool = False, max_traj: int = 50):
        super().__init__()

        data_path = CURRENT_DIR / "data"
        model_path = CURRENT_DIR / "models/tinyphysics.onnx"

        # Load the model
        self.model = TinyPhysicsModel(str(model_path), debug)
        self.debug = debug
        self.datasets = sorted(list(data_path.glob("*.csv")))[:max_traj]

        # Define action space and observation space
        self.max_range = max_range
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # TODO: maybe include history?
        n_state = 3
        n_target = 1
        n_pid = 4
        # n_obs = n_state + n_target + FUTURE_PLAN_STEPS * n_state + n_pid
        n_obs = n_pid

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

    @staticmethod
    def get_observation(
        state: State,
        current_lataccel: float,
        target: float,
        future_plan: FuturePlan,
        last_error: float,
        error_integral: float,
    ) -> np.ndarray:
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
        current_error = target - current_lataccel
        error_diff = current_error - last_error
        pid_obs = np.array(
            [
                current_lataccel / MAX_LATACCEL,
                current_error,
                error_diff,
                error_integral,
            ]
        )
        return pid_obs.astype(np.float32).flatten()
        # Concatenate all observations
        # return np.concatenate([state_obs, target_obs, future_plan_obs, pid_obs]).astype(np.float32)

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
            self.last_error = target - current_lataccel
            self.error_integral += self.last_error
            self.step_idx += 1

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

        jerk_penalty = 0.0
        if self.step_idx > CONTROL_START_IDX:
            # TODO: check to penalize action jerk instead
            jerk_penalty = (self.current_lataccel_history[-1] - self.current_lataccel_history[-2]) ** 2

        last_target = self.target_lataccel_history[-1]
        # Calculate reward
        tracking_penalty = -((last_target - current_lataccel) ** 2) / MAX_LATACCEL * LAT_ACCEL_COST_MULTIPLIER
        jerk_penalty = -jerk_penalty / MAX_JERK

        # print(f"tracking_penalty: {tracking_penalty:>6.4}, jerk_penalty: {jerk_penalty:>6.4}")
        reward = tracking_penalty + jerk_penalty

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
        self.error_integral += last_target - current_lataccel
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
        self.last_error = last_target - current_lataccel

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
