import argparse
import importlib
import os
import signal
import urllib.request
import zipfile
from collections import namedtuple
from functools import partial
from hashlib import md5
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
from gymnasium.spaces import Box
from tqdm.contrib.concurrent import process_map

from controllers import BaseController
from controllers.pid import Controller as PIDController

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

CURRENT_DIR = Path(__file__).resolve().parent

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple("State", ["roll_lataccel", "v_ego", "a_ego"])
FuturePlan = namedtuple("FuturePlan", ["lataccel", "roll_lataccel", "v_ego", "a_ego"])

DATASET_URL = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
DATASET_PATH = Path(__file__).resolve().parent / "data"


class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value: Union[float, np.ndarray, list[float]]) -> Union[int, np.ndarray]:
        value = self.clip(value)
        return np.digitize(value, self.bins, right=True)

    def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return self.bins[token]

    def clip(self, value: Union[float, np.ndarray, list[float]]) -> Union[float, np.ndarray]:
        return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
    def __init__(self, model_path: str, debug: bool) -> None:
        self.tokenizer = LataccelTokenizer()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = "CPUExecutionProvider"

        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, [provider])

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=1.0) -> int:
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        # we only care about the last timestep (batch size is just 1)
        assert probs.shape[0] == 1
        assert probs.shape[2] == VOCAB_SIZE
        sample = np.random.choice(probs.shape[2], p=probs[0, -1])
        return sample

    def get_current_lataccel(
        self,
        sim_states: list[State],
        actions: list[float],
        past_preds: list[float],
    ) -> float:
        tokenized_actions = self.tokenizer.encode(past_preds)
        raw_states = [list(x) for x in sim_states]
        states = np.column_stack([actions, raw_states])
        input_data = {
            "states": np.expand_dims(states, axis=0).astype(np.float32),
            "tokens": np.expand_dims(tokenized_actions, axis=0).astype(np.int64),
        }
        return self.tokenizer.decode(self.predict(input_data, temperature=0.8))  # type: ignore[return-value]


class TinyPhysicsSimulator:
    def __init__(
        self,
        model: TinyPhysicsModel,
        data_path: str,
        controller: Optional[BaseController] = None,
        debug: bool = False,
        reset: bool = True,
    ) -> None:
        self.data_path = data_path
        self.sim_model = model
        self.data = self.get_data(data_path)
        self.controller = controller
        self.debug = debug
        if reset:
            self.reset()

    def reset(self, fixed_seed: bool = True) -> None:
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = [x[0] for x in state_target_futureplans]
        self.action_history = self.data["steer_command"].values[: self.step_idx].tolist()
        self.current_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_future = None
        self.current_lataccel = self.current_lataccel_history[-1]
        if fixed_seed:
            seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
            np.random.seed(seed)

    def get_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        processed_df = pd.DataFrame(
            {
                "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
                "v_ego": df["vEgo"].values,
                "a_ego": df["aEgo"].values,
                "target_lataccel": df["targetLateralAcceleration"].values,
                # steer commands are logged with left-positive convention but this simulator uses right-positive
                "steer_command": -df["steerCommand"].values,
            }
        )
        return processed_df

    def sim_step(self, step_idx: int) -> None:
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:],
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

        self.current_lataccel_history.append(self.current_lataccel)

    def control_step(self, step_idx: int, action: Optional[float]) -> None:
        if action is None and not step_idx < CONTROL_START_IDX:
            assert self.controller is not None, "Controller is not initialized"
            action = self.controller.update(
                self.target_lataccel_history[step_idx],
                self.current_lataccel,
                self.state_history[step_idx],
                future_plan=self.futureplan,
            )
        if step_idx < CONTROL_START_IDX:
            action = self.data["steer_command"].values[step_idx]
        assert action is not None
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history.append(action)

    def get_observation(self) -> np.ndarray:
        state, target_lataccel, futureplan = self.get_state_target_futureplan(self.step_idx)

        # For the last step, no future plan is available
        if self.step_idx == len(self.data) - 1:
            futureplan = FuturePlan(
                lataccel=[0] * FUTURE_PLAN_STEPS,
                roll_lataccel=[0] * FUTURE_PLAN_STEPS,
                v_ego=[0] * FUTURE_PLAN_STEPS,
                a_ego=[0] * FUTURE_PLAN_STEPS,
            )

        return (
            np.array(
                [
                    self.current_lataccel,
                    target_lataccel,
                    state.roll_lataccel,
                    state.v_ego,
                    state.a_ego,
                    # TODO: give more of the future plan
                    futureplan.lataccel[0],
                    futureplan.roll_lataccel[0],
                    futureplan.v_ego[0],
                    futureplan.a_ego[0],
                ]
            )
            .flatten()
            .astype(np.float32)
        )

    def get_state_target_futureplan(self, step_idx: int) -> tuple[State, float, FuturePlan]:
        state = self.data.iloc[step_idx]
        return (
            State(roll_lataccel=state["roll_lataccel"], v_ego=state["v_ego"], a_ego=state["a_ego"]),
            state["target_lataccel"],
            FuturePlan(
                lataccel=self.data["target_lataccel"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
                roll_lataccel=self.data["roll_lataccel"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
                v_ego=self.data["v_ego"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
                a_ego=self.data["a_ego"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            ),
        )

    def step(self, action: Optional[float] = None) -> tuple[np.ndarray, float, bool, bool, dict]:
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.futureplan = futureplan
        self.control_step(self.step_idx, action)
        self.sim_step(self.step_idx)
        self.step_idx += 1
        if self.step_idx < len(self.data):
            obs = self.get_observation()
            reward = self.compute_immediate_reward()
        else:
            obs = np.zeros(9)
            reward = 0.0

        terminated = False
        truncated = self.step_idx >= len(self.data) - 1
        return obs, reward, terminated, truncated, {}

    def compute_immediate_reward(self) -> float:
        last_pred = np.array(self.current_lataccel_history)[-2]
        pred = np.array(self.current_lataccel_history)[-1]
        target = np.array(self.target_lataccel_history)[-1]
        lat_accel_cost = np.mean((target - pred) ** 2)
        # jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
        jerk_cost = np.mean(((pred - last_pred) / DEL_T) ** 2)
        jerk_cost = np.clip(jerk_cost, 0, 5)
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return -total_cost * 0.1

    def plot_data(self, ax, lines, axis_labels, title) -> None:
        ax.clear()
        for line, label in lines:
            ax.plot(line, label=label)
        ax.axline(
            (CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color="black", linestyle="--", alpha=0.5, label="Control Start"
        )
        ax.legend()
        ax.set_title(f"{title} | Step: {self.step_idx}")
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    def compute_cost(self) -> dict[str, float]:
        target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

        lat_accel_cost = np.mean((target - pred) ** 2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return {"lataccel_cost": lat_accel_cost, "jerk_cost": jerk_cost, "total_cost": total_cost}

    def rollout(self) -> dict[str, float]:
        if self.debug:
            plt.ion()
            fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

        for _ in range(CONTEXT_LENGTH, len(self.data)):
            self.step()
            if self.debug and self.step_idx % 10 == 0:
                print(
                    f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}"
                    f", Target lataccel: {self.target_lataccel_history[-1]:>6.2f}"
                )
                self.plot_data(
                    ax[0],
                    [(self.target_lataccel_history, "Target lataccel"), (self.current_lataccel_history, "Current lataccel")],
                    ["Step", "Lateral Acceleration"],
                    "Lateral Acceleration",
                )
                self.plot_data(ax[1], [(self.action_history, "Action")], ["Step", "Action"], "Action")
                self.plot_data(
                    ax[2],
                    [(np.array(self.state_history)[:, 0], "Roll Lateral Acceleration")],
                    ["Step", "Lateral Accel due to Road Roll"],
                    "Lateral Accel due to Road Roll",
                )
                self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], "v_ego")], ["Step", "v_ego"], "v_ego")
                plt.pause(0.01)

        if self.debug:
            plt.ioff()
            plt.show()
        return self.compute_cost()


# Gym interface for the simulator
class TinyPhysicsEnv(gymnasium.Env):
    sim: TinyPhysicsSimulator

    def __init__(self, debug: bool = False, max_range: float = 2.0, use_pid: bool = False, pid_actions: bool = False):
        super().__init__()
        self.data_path = CURRENT_DIR / "data"
        # Do not take the first 7000 files, used for evaluation
        self.datasets = sorted(list(self.data_path.glob("*.csv")))[7000:]
        self.model_path = CURRENT_DIR / "models/tinyphysics.onnx"

        # TODO: check bounds, should be -5 to 5
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # with future plan (next target lataccel, next roll lataccel, next v_ego, next a_ego)
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # if use_pid:
        #     self.n_obs += 1
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # PID obs
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        # PID obs + future plan + last action
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.pid_actions = pid_actions
        if pid_actions:
            # Kp, Ki, Kd
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.action_scale = max_range  # STEER_RANGE[1]
        self.pid_controller = PIDController()
        self.use_pid = use_pid
        self.prev_error = 0.0
        self.error_integral = 0.0

        self.debug = debug

    def reset(self, seed: Optional[int] = None, options=None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        # Select a random dataset
        data_path = self.data_path / np.random.choice(self.datasets)  # type: ignore[arg-type]

        self.sim = TinyPhysicsSimulator(
            TinyPhysicsModel(str(self.model_path), self.debug), str(data_path), None, self.debug, reset=False
        )
        # self.sim.controller = self.pid_controller
        self.sim.reset(fixed_seed=False)
        # Warm up the controller
        for _ in range(CONTROL_START_IDX - CONTEXT_LENGTH):
            obs, _, _, _, _ = self.sim.step()

        # if self.use_pid:
        #     obs = np.append(obs, 0.0)
        _, target, _ = self.sim.get_state_target_futureplan(self.sim.step_idx)

        # PID obs
        self.error_integral = 0.0
        self.prev_error = 0.0
        obs = self.get_pid_plus_obs(self.sim.current_lataccel, target)

        return obs, {}

    def get_pid_obs(self, current_lataccel: float, target_lataccel: float) -> np.ndarray:
        error = target_lataccel - current_lataccel
        error_diff = error - self.prev_error
        self.error_integral += error
        self.prev_error = error
        self.error_integral = np.clip(self.error_integral, -5, 5)
        last_action = self.sim.action_history[-1] if self.sim.action_history else 0.0
        return np.array([error, error_diff, self.error_integral, last_action]).astype(np.float32)

    def get_pid_plus_obs(self, current_lataccel: float, target_lataccel: float) -> np.ndarray:
        obs_pid = self.get_pid_obs(current_lataccel, target_lataccel)
        _, _, futureplan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        next_lateral_accel = np.zeros(3)
        n_next = min(len(futureplan.lataccel), 3)
        next_lateral_accel[:n_next] = target_lataccel
        next_lateral_accel[:n_next] -= futureplan.lataccel[:n_next]
        return np.array([*obs_pid, *next_lateral_accel]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # TODO: check off-by-one error
        state, target, futureplan = self.sim.get_state_target_futureplan(self.sim.step_idx)

        if self.pid_actions:
            kp, ki, kd = action
            self.pid_controller.p = abs(kp)
            self.pid_controller.i = abs(ki) * 0.1
            self.pid_controller.d = -abs(kd) * 0.5
            action = self.pid_controller.update(  # type: ignore[assignment]
                target,
                self.sim.current_lataccel,
                state,
                future_plan=futureplan,
            )
        else:
            action = action.item() * self.action_scale

        if self.use_pid:
            action_pid = self.pid_controller.update(
                target,
                self.sim.current_lataccel,
                state,
                future_plan=futureplan,
            )
            action += action_pid
            # action = action_pid

        obs, reward, terminated, truncated, info = self.sim.step(float(action))

        _, target, _ = self.sim.get_state_target_futureplan(self.sim.step_idx)

        obs = self.get_pid_plus_obs(self.sim.current_lataccel, target)

        # if self.use_pid:
        #     # TODO: maybe add state of PID
        #     obs = np.append(obs, action_pid)

        if truncated:
            costs = self.sim.compute_cost()
            print(
                f"lataccel_cost: {costs['lataccel_cost']:>6.4}, jerk_cost: {costs['jerk_cost']:>6.4}, "
                f"total_cost: {costs['total_cost']:>6.4}"
            )

        return obs, reward, terminated, truncated, info


def get_available_controllers():
    return [f.stem for f in Path("controllers").iterdir() if f.is_file() and f.suffix == ".py" and f.stem != "__init__"]


def run_rollout(data_path, controller_type, model_path, debug=False):
    tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
    controller = importlib.import_module(f"controllers.{controller_type}").Controller()
    sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=debug)
    return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history


def download_dataset():
    print("Downloading dataset (0.6G)...")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL) as resp:
        with zipfile.ZipFile(BytesIO(resp.read())) as z:
            for member in z.namelist():
                if not member.endswith("/"):
                    with z.open(member) as src, open(DATASET_PATH / os.path.basename(member), "wb") as dest:
                        dest.write(src.read())


if __name__ == "__main__":
    available_controllers = get_available_controllers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--controller", default="pid", choices=available_controllers)
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        download_dataset()

    data_path = Path(args.data_path)
    if data_path.is_file():
        cost, _, _ = run_rollout(data_path, args.controller, args.model_path, debug=args.debug)
        print(
            f"\nAverage lataccel_cost: {cost['lataccel_cost']:>6.4}, "
            f"average jerk_cost: {cost['jerk_cost']:>6.4}, average total_cost: {cost['total_cost']:>6.4}"
        )
    elif data_path.is_dir():
        run_rollout_partial = partial(run_rollout, controller_type=args.controller, model_path=args.model_path, debug=False)
        files = sorted(data_path.iterdir())[: args.num_segs]
        results = process_map(run_rollout_partial, files, max_workers=16, chunksize=10)
        costs = [result[0] for result in results]
        costs_df = pd.DataFrame(costs)
        print(
            f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, "
            f"average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, "
            f"average total_cost: {np.mean(costs_df['total_cost']):>6.4}"
        )
        for cost in costs_df.columns:
            plt.hist(costs_df[cost], bins=np.arange(0, 1000, 10), label=cost, alpha=0.5)  # type: ignore
        plt.xlabel("costs")
        plt.ylabel("Frequency")
        plt.title("costs Distribution")
        plt.legend()
        plt.show()
