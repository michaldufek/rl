from gym.envs.registration import register

register(
    id='stockpicker',
    entry_point='stockpicer_env.envs:StockPickerEnv',
)