default_hyperparams = dict(
    n_envs=1,
    n_timesteps=int(1e6),
    policy="MlpPolicy",
    # qf_learning_rate=1e-3,
    # CrossQ + DroQ
    # policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[1024, 1024]), dropout_rate=0.01),
    policy_kwargs=dict(layer_norm=True, dropout_rate=0.01),
    gradient_steps=10,
    policy_delay=10,
    learning_starts=5000,
    env_wrapper=[
        # {"rl_zoo3.wrappers.HistoryWrapper": {"horizon": 2}},
        {
            "custom_envs.filter_wrappers.ActionFilterWrapper": {
                "sampling_rate": 60,
                "lowcut": 0,
                "highcut": 4,
            }
        },
    ],
    # normalize={"norm_obs": True, "norm_reward": False},
)

hyperparams = {}

for env_id in [
    "TinyPhysicsEnv-v0",
]:
    hyperparams[env_id] = default_hyperparams
