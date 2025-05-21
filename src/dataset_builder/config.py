import envyaml
import os
from dotenv import load_dotenv

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

def load_config(config_file, preset="default", config_dir=CONFIG_DIR):
    load_dotenv()
    cfg = envyaml.EnvYAML(os.path.join(config_dir, config_file), strict=False)[preset]
    return cfg

CONFIG = load_config("config.yaml")
