"""
Interface for user configurations.

Reads configurations from a YAML file, which may be located in the following locations (in order of precedence):
1. Path specified by the 'ODDEEG_USER_CONFIG' environment variable.
2. `user_config.yaml` file in the current working directory.
3. `user_config.yaml` file in the parent directory of this file.

Returns a dictionary with the loaded configurations.

Example usage:
    from oddeeg.user_config import user_config
    print(user_config)
"""

import os
from pathlib import Path

import yaml


def load_user_config(config_path: str | Path = None) -> dict:
    """Load user configuration from a YAML file.

    Args:
        config_path (str | Path, optional): Path to the configuration file.
            If None, searches in predefined locations (in order of precedence):
            1. Path specified by the 'ODDEEG_USER_CONFIG' environment variable.
            2. `user_config.yaml` file in the current working directory.
            3. `user_config.yaml` file in the repository root (two levels up from this file).

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    if config_path is not None:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Configuration file not found at specified path: {config_path}") from err
    else:
        # 1. Check environment variable
        env_config_path = os.getenv("ODDEEG_USER_CONFIG")
        if env_config_path and Path(env_config_path).is_file():
            with open(env_config_path) as f:
                config = yaml.safe_load(f)
            return config
        # 2. Check current working directory
        cwd_config_path = Path.cwd() / "user_config.yaml"
        if cwd_config_path.is_file():
            with open(cwd_config_path) as f:
                config = yaml.safe_load(f)
            return config
        # 3. Check repository root (two levels up from this file)
        repo_root_config_path = Path(__file__).parent.parent.parent / "user_config.yaml"
        if repo_root_config_path.is_file():
            with open(repo_root_config_path) as f:
                config = yaml.safe_load(f)
            return config
        # Raise error if no configuration file is found
        raise FileNotFoundError(
            "No user configuration file found in any of the predefined locations: "
            f"{[env_config_path, cwd_config_path, repo_root_config_path]}"
        )


user_config = load_user_config()
