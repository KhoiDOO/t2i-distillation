import os
import shutil

from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print('Found no save directory: ./results')
        exit(0)

    extract_dir = 'extract_results'
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    shutil.copytree(results_dir, extract_dir)

    exp_dirs = glob(f'{extract_dir}/*/*/*/*')

    for exp_dir in tqdm(exp_dirs):
        debug_dir = os.path.join(exp_dir, 'debug')

        if not os.path.exists(debug_dir):
            shutil.rmtree(exp_dir)
            continue
        elif len(os.listdir(debug_dir)) < 5:
            shutil.rmtree(exp_dir)
            continue
        else:
            shutil.rmtree(debug_dir)
        
        config_path = os.path.join(exp_dir, 'config.json')
        stats_path = os.path.join(exp_dir, 'stats.json')
        video_path = os.path.join(exp_dir, 'debug_optimization.mp4')

        if os.path.exists(config_path):
            os.remove(config_path)
        
        if os.path.exists(stats_path):
            os.remove(stats_path)
        
        if os.path.exists(video_path):
            os.remove(video_path)