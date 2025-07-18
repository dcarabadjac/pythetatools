import yaml
from types import SimpleNamespace
import os

CONFIG = None
CONFIG_KEYS = []

library_dir = os.path.dirname(os.path.abspath(__file__))
inputs_dir = os.path.join(os.path.dirname(library_dir), 'inputs')
outputs_dir = os.path.join(os.path.dirname(library_dir), 'outputs')

def load_config(path):
    global CONFIG_PATH, CONFIG, CONFIG_KEYS
    CONFIG_PATH = path
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    CONFIG = SimpleNamespace(**data)
    CONFIG_KEYS = list(data.keys())

