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
project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir))
print(f"Project directory added to sys.path: {project_dir}")

log = logging

# --- Import from the CORRECT 'tldw_app' package ---
try:
    # Use 'tldw_app' consistently
    from tldw_app.app import TldwCli
    from tldw_app.config import load_config, get_config_path, DEFAULT_CONFIG
    # Ensure this import path is correct based on your structure inside tldw_app
    from tldw_app.css.default_css import DEFAULT_CSS_CONTENT
except ModuleNotFoundError as e:
    # Update the error message to reflect 'tldw_app'
    print(f"ERROR: run.py: Failed to import from tldw_app package.")
    print(f"       Ensure '{project_dir}' is correct and contains 'tldw_app'.") # Check for tldw_app
    print(f"       Make sure tldw_app and its subdirs have __init__.py files.")
    print(f"       Original error: {e}")
    sys.exit(1)

def ensure_default_files():
    """Creates default config and CSS files if they don't exist."""
    # Config File
    config_path = get_config_path()
    if not config_path.exists():
        print(f"Config file not found at {config_path}, creating default.")  # Use print before logging setup
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                toml.dump(DEFAULT_CONFIG, f)
            print(f"Created default configuration file: {config_path}")
        except ImportError:
            print("ERROR: `toml` library not installed. Cannot write default config.")
        except Exception as e:
            print(f"ERROR: Failed to create default config file: {e}")

    # CSS File - Check if app.py handles its CSS path correctly now
    try:
        css_file = project_dir / "tldw_app" / TldwCli.CSS_PATH
        if not css_file.is_file():
            print(f"CSS file not found at {css_file}, creating default.")
            css_file.parent.mkdir(parents=True, exist_ok=True)
            # Make sure DEFAULT_CSS_CONTENT is defined or loaded correctly
            with open(css_file, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CSS_CONTENT)  # Assuming this constant exists
            print(f"Created default CSS file: {css_file}")
    except AttributeError:
        print("WARNING: Could not determine CSS_PATH from TldwCli class. Skipping CSS check.")
    except NameError:
        print("WARNING: DEFAULT_CSS_CONTENT not defined. Skipping CSS creation.")  # Handle missing constant
    except Exception as e:
        print(f"ERROR: Failed to create default CSS file: {e}")


if __name__ == "__main__":
    print("Ensuring default files...")
    ensure_default_files()

    print("Starting tldw-cli application...")
    # Initialize logger *inside* the app's lifecycle (on_mount)
    app = TldwCli()
    app.run()
    print("tldw-cli application finished.")

#
# End of run.py
#######################################################################################################################
