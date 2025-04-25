# run.py
# Description: This file is the entry point for the tldw-cli application. It sets up the environment, ensures necessary files exist, and runs the application.
#
# Imports
import logging
from pathlib import Path
import sys

import toml

# 3rd-party Libraries
#
# Local Imports
# --- Add project root to sys.path ---
# This allows Python to find the 'tldw_app' package
project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir))
print(f"Project directory added to sys.path: {project_dir}")

# --- Import from the 'tldw_app' package ---
try:
    from app import TldwCli
    from config import load_config, get_config_path, DEFAULT_CONFIG
    # Assuming you move the CSS string content to a file like tldw_app/css/default_css.py
    from CSS.default_css import DEFAULT_CSS_CONTENT
except ModuleNotFoundError as e:
    print(f"ERROR: Failed to import from tldw_app package.")
    print(f"       Ensure '{project_dir}' is correct and contains 'tldw_app'.")
    print(f"       Make sure tldw_app and its subdirs have __init__.py files.")
    print(f"       Original error: {e}")
    sys.exit(1) # Exit if core components can't be imported

# --- Initial Setup ---
logging.basicConfig(level="DEBUG", format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger(__name__)

def ensure_default_files():
    """Creates default config and CSS files if they don't exist."""
    # Config File
    config_path = get_config_path() # This function likely looks in ~/.config, which is fine
    if not config_path.exists():
        log.warning(f"Config file not found at {config_path}, creating default.")
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                toml.dump(DEFAULT_CONFIG, f)
            log.info(f"Created default configuration file: {config_path}")
        except ImportError:
            log.error("`toml` library not installed. Cannot write default config. Please install it (`pip install toml`).")
        except Exception as e:
            log.error(f"Failed to create default config file: {e}", exc_info=True)

    # CSS File - Path needs to point *inside* the package now
    # TldwCli.CSS_PATH should be relative e.g., "css/tldw_cli.tcss"
    try:
        # Construct path relative to the project root where run.py is
        css_path_in_package = project_dir / "tldw_app" / TldwCli.CSS_PATH
        if not css_path_in_package.exists():
            log.warning(f"CSS file not found at {css_path_in_package}, creating default.")
            css_path_in_package.parent.mkdir(parents=True, exist_ok=True)
            with open(css_path_in_package, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CSS_CONTENT)
            log.info(f"Created default CSS file: {css_path_in_package}")
    except AttributeError:
         log.error("Could not determine CSS_PATH from TldwCli class. Skipping CSS check.")
    except Exception as e:
        log.error(f"Failed to create default CSS file: {e}", exc_info=True)


if __name__ == "__main__":
    log.info("Ensuring default files...")
    ensure_default_files()

    log.info("Starting tldw-cli application...")
    app = TldwCli()
    app.run()
    log.info("tldw-cli application finished.")
#
# End of run.py
#######################################################################################################################
