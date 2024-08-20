from collections import namedtuple
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort
import pandas as pd

from controllers.pid import Controller as PIDController

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


class LataccelTokenizer:
    vocab_size: int = 1024
    lat_accell_min: int = -5
    lat_accell_max: int = 5

    def __init__(self):
        self.bins = np.linspace(self.lat_accell_min, self.lat_accell_max, self.vocab_size)

    def encode(self, value: Union[float, np.ndarray, list[float]]) -> Union[int, np.ndarray]:
        value = np.clip(value, self.lat_accell_min, self.lat_accell_max)
        return np.digitize(value, self.bins, right=True)

    def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return self.bins[token]


class TinyPhysicsModel:
    def __init__(self, model_path: str, debug: bool) -> None:
        self.tokenizer = LataccelTokenizer()
        # Force prediction on CPU
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = "CPUExecutionProvider"

        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, [provider])

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        # Stable softmax
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature: float = 1.0) -> int:
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        # we only care about the last timestep (batch size is just 1)
        # assert probs.shape[0] == 1
        # assert probs.shape[2] == VOCAB_SIZE
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


def get_data(data_path: Path) -> pd.DataFrame:
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


def get_state_target_futureplan(dataframe: pd.DataFrame, step_idx: int) -> tuple[State, float, FuturePlan]:
    state = data.iloc[step_idx]
    return (
        State(roll_lataccel=state["roll_lataccel"], v_ego=state["v_ego"], a_ego=state["a_ego"]),
        state["target_lataccel"],
        FuturePlan(
            lataccel=data["target_lataccel"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=data["roll_lataccel"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            v_ego=data["v_ego"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            a_ego=data["a_ego"].values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
        ),
    )


if __name__ == "__main__":
    n_trajectories = 5
    debug = False
    data_path = CURRENT_DIR / "data"
    # Do not take the first 7000 files, used for evaluation
    datasets = sorted(list(data_path.glob("*.csv")))[:50]
    model_path = CURRENT_DIR / "models/tinyphysics.onnx"
    sim_model = TinyPhysicsModel(str(model_path), debug)

    np.random.seed(0)

    for _ in range(n_trajectories):
        data_path = data_path / np.random.choice(datasets)  # type: ignore[arg-type]

        print(data_path.name)
        data = get_data(data_path)
        controller = PIDController()

        step_idx = CONTEXT_LENGTH
        state_target_futureplans = [get_state_target_futureplan(data, i) for i in range(step_idx)]
        state_history = [x[0] for x in state_target_futureplans]
        action_history = data["steer_command"].values[:step_idx].tolist()
        current_lataccel_history = [x[1] for x in state_target_futureplans]
        target_lataccel_history = [x[1] for x in state_target_futureplans]
        target_future = None
        current_lataccel = current_lataccel_history[-1]

        for step_idx in range(CONTEXT_LENGTH, len(data)):
            state, target, futureplan = get_state_target_futureplan(data, step_idx)
            state_history.append(state)
            target_lataccel_history.append(target)
            # control_step(step_idx)
            action = controller.update(
                target_lataccel_history[step_idx],
                current_lataccel,
                state_history[step_idx],
                future_plan=futureplan,
            )
            if step_idx < CONTROL_START_IDX:
                action = data["steer_command"].values[step_idx]

            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            action_history.append(action)

            # sim_step(step_idx)
            pred = sim_model.get_current_lataccel(
                sim_states=state_history[-CONTEXT_LENGTH:],
                actions=action_history[-CONTEXT_LENGTH:],
                past_preds=current_lataccel_history[-CONTEXT_LENGTH:],
            )

            pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            if step_idx >= CONTROL_START_IDX:
                current_lataccel = pred
            else:
                current_lataccel = get_state_target_futureplan(data, step_idx)[1]

            current_lataccel_history.append(current_lataccel)

        # Compute cost
        targets = np.array(target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        predictions = np.array(current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

        lat_accel_cost = np.mean((targets - predictions) ** 2) * 100
        jerk_cost = np.mean((np.diff(predictions) / DEL_T) ** 2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost

        print(f"lataccel_cost: {lat_accel_cost:>6.4}, jerk_cost: {jerk_cost:>6.4}, total_cost: {total_cost:>6.4}")
