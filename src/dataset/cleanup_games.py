import os
import shutil

from util.config import DATA_DIRECTORY

for phase in ["train", "validate", "test"]:
    dir = os.path.join(DATA_DIRECTORY, phase)
    if os.path.exists(dir):
        shutil.rmtree(dir)
