# import flax.linen as nn

# Default hyperparameters for SB3 are tuned for MuJoCo
default_hyperparams = dict(
    n_envs=1,
    n_timesteps=int(1e6),
    policy="MlpPolicy",
    # policy_kwargs={"log_std_init": -1},
    # policy_kwargs=dict(
    #     # activation_fn=nn.relu,
    #     # log_std_init=0.0,
    #     net_arch=[256, 256],
    # ),
    normalize=True,
    # env_wrapper=[{"rl_zoo3.wrappers.HistoryWrapper": {"horizon": 2}}],
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
