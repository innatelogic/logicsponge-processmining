import copy
from typing import Any

from logicsponge.processmining.types import Config

# Centralized configuration instance
DEFAULT_CONFIG: Config = {
    "start_symbol": "__start__",
    "stop_symbol": "__stop__",
    "discount_factor": 0.9,
    "randomized": False,
    "top_k": 3,
    "include_stop": True,
}


def update_config(custom_param: dict[str, Any] | None = None) -> Config:
    """
    Merge custom configuration with defaults, returning a new dictionary.
    :param custom_param: Optional dictionary with configuration overrides.
    :return: Merged configuration as a new dictionary.
    """
    default_config = DEFAULT_CONFIG
    updated_config = copy.deepcopy(default_config)  # Create a deep copy of the default config

    if custom_param:
        for key, value in custom_param.items():
            if key not in updated_config:
                msg = f"Invalid configuration key: {key}"
                raise KeyError(msg)
            updated_config[key] = value  # Apply custom overrides

    return updated_config
