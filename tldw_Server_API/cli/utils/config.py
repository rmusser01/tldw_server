"""
Configuration management utilities for CLI.

Handles loading and validating configuration files with proper
error handling and defaults.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


def load_cli_config(config_path: Optional[str] = None, db_path_override: Optional[str] = None) -> dict[str, Any]:
    """
    Load CLI configuration from various sources.

    Args:
        config_path: Explicit path to config file
        db_path_override: Override database path

    Returns:
        Configuration dictionary

    Raises:
        ConfigError: If configuration cannot be loaded or is invalid
    """
    config = {}

    # Load from evaluations config file
    eval_config_path = _find_evaluations_config(config_path)
    if eval_config_path and eval_config_path.exists():
        try:
            with open(eval_config_path) as f:
                eval_config = yaml.safe_load(f)
                if eval_config:
                    config.update(eval_config)
                    logger.debug(f"Loaded evaluations config from {eval_config_path}")
        except Exception as e:
            raise ConfigError(f"Failed to load evaluations config from {eval_config_path}: {e}") from e

    # Load from main tldw config file
    main_config_path = _find_main_config()
    if main_config_path and main_config_path.exists():
        try:
            from tldw_Server_API.app.core.config import load_comprehensive_config
            main_config = load_comprehensive_config()
            if main_config:
                # Convert ConfigParser to dict for easier handling
                main_dict = {}
                for section in main_config.sections():
                    main_dict[section.lower()] = dict(main_config[section])

                # Merge main config (evaluations config takes precedence)
                for key, value in main_dict.items():
                    if key not in config:
                        config[key] = value

                logger.debug(f"Loaded main config from {main_config_path}")
        except Exception as e:
            logger.warning(f"Could not load main config: {e}")

    # Apply overrides
    if db_path_override:
        config.setdefault('database', {})
        config['database']['path'] = db_path_override

    # Set defaults
    config = _apply_defaults(config)

    # Validate configuration
    _validate_config(config)

    return config


def _find_evaluations_config(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Find evaluations configuration file."""
    if explicit_path:
        return Path(explicit_path)

    # Search locations in order of preference
    search_paths = [
        Path.cwd() / "evaluations_config.yaml",
        Path.cwd() / "config" / "evaluations_config.yaml",
        Path(__file__).parent.parent.parent / "Config_Files" / "evaluations_config.yaml",
    ]

    # Add environment variable path
    env_config = os.getenv("TLDW_EVALS_CONFIG")
    if env_config:
        search_paths.insert(0, Path(env_config))

    for path in search_paths:
        if path.exists() and path.is_file():
            return path

    logger.debug("No evaluations config file found in standard locations")
    return None


def _find_main_config() -> Optional[Path]:
    """Find main tldw configuration file."""
    search_paths = [
        Path.cwd() / "config.txt",
        Path(__file__).parent.parent.parent / "Config_Files" / "config.txt",
    ]

    # Add environment variable path
    env_config = os.getenv("TLDW_CONFIG_PATH")
    if env_config:
        search_paths.insert(0, Path(env_config))

    for path in search_paths:
        if path.exists() and path.is_file():
            return path

    logger.debug("No main config file found")
    return None


def _apply_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply default configuration values."""
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP
    defaults = {
        'database': {
            # Default to per-user evaluations DB (single-user ID)
            'path': str(_DP.get_evaluations_db_path(_DP.get_single_user_id())),
            'connection': {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600
            }
        },
        'rate_limiting': {
            'global': {
                'default_tier': 'free'
            },
            'tiers': {
                'free': {
                    'evaluations_per_minute': 10,
                    'evaluations_per_day': 100,
                    'total_tokens_per_day': 100000,
                    'burst_size': 5
                }
            }
        },
        'webhooks': {
            'delivery': {
                'max_retries': 3,
                'timeout_seconds': 30
            },
            'security': {
                'require_https': False,
                'validate_ssl_certificates': True
            }
        },
        'security': {
            'validation': {
                'max_text_length': 100000,
                'max_batch_size': 100
            },
            'audit': {
                'log_all_requests': True,
                'retention_days': 90
            }
        },
        'monitoring': {
            'metrics': {
                'enabled': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
    }

    # Deep merge defaults with config
    return _deep_merge(defaults, config)


def _deep_merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = default.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _validate_config(config: dict[str, Any]):
    """Validate configuration structure and values."""
    errors = []

    # Validate database configuration
    if 'database' not in config:
        errors.append("Missing 'database' section")
    else:
        db_config = config['database']
        if 'path' not in db_config:
            errors.append("Missing 'database.path'")
        else:
            # Ensure database directory exists
            db_path = Path(db_config['path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate rate limiting configuration
    if 'rate_limiting' in config:
        rate_config = config['rate_limiting']
        if 'tiers' in rate_config:
            for tier_name, tier_data in rate_config['tiers'].items():
                if not isinstance(tier_data, dict):
                    errors.append(f"Invalid tier configuration: {tier_name}")
                    continue

                required_fields = [
                    'evaluations_per_minute',
                    'evaluations_per_day'
                ]

                for field in required_fields:
                    if field not in tier_data:
                        errors.append(f"Missing field '{field}' in tier '{tier_name}'")
                    elif not isinstance(tier_data[field], (int, float)):
                        errors.append(f"Invalid type for '{field}' in tier '{tier_name}' (expected number)")

    if errors:
        raise ConfigError(f"Configuration validation failed: {'; '.join(errors)}")


def get_database_path(config: dict[str, Any]) -> str:
    """Get database path from configuration."""
    return config.get('database', {}).get('path', 'evaluations.db')


def get_log_level(config: dict[str, Any]) -> str:
    """Get logging level from configuration."""
    return config.get('monitoring', {}).get('logging', {}).get('level', 'INFO')


def get_config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get configuration value by dot-separated path.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'rate_limiting.tiers.free.evaluations_per_minute')
        default: Default value if path not found

    Returns:
        Configuration value or default
    """
    current = config

    for key in path.split('.'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def save_config(config: dict[str, Any], config_path: Optional[str] = None):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save
        config_path: Path to save config file (defaults to evaluations_config.yaml)
    """
    if not config_path:
        config_path = _find_evaluations_config()
        if not config_path:
            config_path = Path.cwd() / "evaluations_config.yaml"
    else:
        config_path = Path(config_path)

    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)

        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        raise ConfigError(f"Failed to save configuration to {config_path}: {e}") from e
