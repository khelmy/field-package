import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
ENV_NAME = "Field-v0"

register(
    id = ENV_NAME,
    entry_point = 'field.envs:FieldEnv'
)
