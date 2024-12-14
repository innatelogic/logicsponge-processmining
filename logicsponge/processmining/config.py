import copy
from dataclasses import asdict, dataclass


@dataclass
class DefaultConfig:
    start_symbol: str = "__start__"
    stop_symbol: str = "__stop__"
    discount_factor: float = 0.9
    randomized: bool = False
    top_k: int = 3
    include_stop: bool = True


# Centralized configuration instance
DEFAULT_CONFIG = asdict(DefaultConfig())


def update_config(custom_config: dict | None = None) -> dict:
    """
    Merge custom configuration with defaults, returning a new dictionary.
    :param custom_config: Optional dictionary with configuration overrides.
    :return: Merged configuration as a new dictionary.
    """
    default_config = DEFAULT_CONFIG
    updated_config = copy.deepcopy(default_config)  # Create a deep copy of the default config

    if custom_config:
        for key, value in custom_config.items():
            if key not in updated_config:
                msg = f"Invalid configuration key: {key}"
                raise KeyError(msg)
            updated_config[key] = value  # Apply custom overrides

    return updated_config
