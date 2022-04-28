from gym.envs.registration import register

register(
    id='meta_envs/Four-room-domain-v0',
    entry_point='meta_envs.envs:Grid',
)