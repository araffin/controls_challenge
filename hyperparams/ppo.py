# Default hyperparameters for SB3 are tuned for MuJoCo
default_hyperparams = dict(
    n_envs=1,
    n_timesteps=int(1e6),
    policy="MlpPolicy",
    policy_kwargs={},
    normalize=True,
    env_wrapper=[{"rl_zoo3.wrappers.HistoryWrapper": {"horizon": 2}}],
)

hyperparams = {}

for env_id in [
    "TinyPhysicsEnv-v0",
]:
    hyperparams[env_id] = default_hyperparams
