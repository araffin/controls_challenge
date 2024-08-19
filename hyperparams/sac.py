default_hyperparams = dict(
    n_envs=1,
    n_timesteps=int(1e6),
    policy="MlpPolicy",
    # qf_learning_rate=1e-3,
    # policy_kwargs={},
    learning_starts=10_000,
    env_wrapper=[{"rl_zoo3.wrappers.HistoryWrapper": {"horizon": 2}}],
)

hyperparams = {}

for env_id in [
    "TinyPhysicsEnv-v0",
]:
    hyperparams[env_id] = default_hyperparams
