default_hyperparams = dict(
    n_envs=1,
    n_timesteps=int(1e6),
    policy="MlpPolicy",
    qf_learning_rate=1e-3,
    # noise_type="ornstein-uhlenbeck",
    noise_type="normal",
    noise_std=0.1,
    batch_size=256,
    normalize={"norm_obs": True, "norm_reward": False},
    # policy_kwargs={},
    learning_starts=5000,
    # env_wrapper=[
    #     # {"rl_zoo3.wrappers.HistoryWrapper": {"horizon": 2}},
    #     {
    #         "custom_envs.filter_wrappers.ActionFilterWrapper": {
    #             "sampling_rate": 60,
    #             "lowcut": 0,
    #             "highcut": 4,
    #         }
    #     },
    # ],
)

hyperparams = {}

for env_id in [
    "TinyPhysicsEnv-v0",
    "LatAccel-v0",
]:
    hyperparams[env_id] = default_hyperparams
