import os
import yaml

from deepmerge import always_merger

class Config(dict):
    """ Application connfigurationa """

    def __init__(self, path=None):
        super().__init__()

        if not os.path.exists(path):
            pass
        else:
            with open(path) as in_file:
                self.update(yaml.safe_load(in_file))

    def __getitem__(self, item):
        return super().__getitem__(item)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CUR_DIR, os.pardir)

DEFAULT_CONFIG = Config(os.path.join(APP_DIR, 'config.yaml'))
LOCAL_CONFIG = Config(os.path.join(APP_DIR, 'config.local.yaml'))

CONFIG = always_merger.merge(DEFAULT_CONFIG, LOCAL_CONFIG)