import os
import shutil

from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    save_dir = f'{os.getcwd()}/results'

    exp_dirs = glob(f'{save_dir}/*/*/*/*')

    for exp_dir in tqdm(exp_dirs):
        debug_dir = os.path.join(exp_dir, 'debug')

        if not os.path.exists(debug_dir):
            shutil.rmtree(exp_dir)
            continue
        elif len(os.listdir(debug_dir)) < 5:
            shutil.rmtree(exp_dir)