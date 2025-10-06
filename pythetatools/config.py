import yaml
from types import SimpleNamespace
import os

# Default configuration
DEFAULT_CONFIG = {
    "my_login": "dcarabad",
    "my_domain": "cca.in2p3.fr",
    "tag": "OARun11A preliminary",
    "dir_ver": "OA2023",
    "oaver": "OA2023",
    "include_320kA": False,
    "data_kind": "data",
    "sample_titles": ['numu1R', 'nue1R', 'numubar1R', 'nuebar1R', 'numucc1pi', 'nue1RD'],
}

CONFIG = SimpleNamespace(**DEFAULT_CONFIG)
CONFIG_KEYS = list(DEFAULT_CONFIG.keys())

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

