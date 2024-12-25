import img2pdf
import os
import json
import shutil
import argparse

from glob import glob
from tqdm import tqdm
from PIL import Image

to_config = {
    'bridge' : ['cfg_scale'],
    'jsdg' : ['numt'],
    'lucid' : ['cfg_scale', 'denoise_cfg_scale', 'deltat', 'deltas'],
    'vsd' : ['cfg_scale'],
    'asd' : ['cfg_scale', 'gamma'],
    'sds' : ['cfg_scale']
}

to_steps = {
    'bridge' : [1000],
    'jsdg' : [5000],
    'lucid' : [1000],
    'vsd' : [1000],
    'asd' : [1000],
    'sds' : [1000]
}

if __name__ == '__main__':
    save_dir = 'results'

    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(save_dir):
        print('Found no save directory: ./results')
        exit(0)
    
    exp_dirs = glob(f'{save_dir}/*/*/*/*')

    for exp_dir in tqdm(exp_dirs):
        debug_dir = os.path.join(exp_dir, 'debug')

        if not os.path.exists(debug_dir):
            continue
        elif len(os.listdir(debug_dir)) < 5:
            continue

        config_path = os.path.join(exp_dir, 'config.json')

        with open(config_path, 'r') as file:
            cfg_dict = json.load(file)
        
        mode = cfg_dict['mode']

        if mode not in to_config:
            continue

        cfgs = to_config[mode]
        steps = to_steps[mode]

        pdf_dir = os.path.join(exp_dir, 'pdfs/visual')

        if os.path.exists(pdf_dir) and args.reset:
            shutil.rmtree(pdf_dir)

        os.makedirs(pdf_dir, exist_ok=True)
        
        for cfg in cfgs:
            for step in steps:
                step_file = os.path.join(debug_dir, f'{step}.png')

                cfg_text = '_'.join([str(cfg_dict[x]) for x in cfgs])

                if not os.path.exists(step_file):
                    continue

                pdf_path = os.path.join(pdf_dir, f'{cfg_text}_{step}.pdf')

                if os.path.exists(pdf_path):
                    continue

                image = Image.open(step_file)

                pdf_bytes = img2pdf.convert(image.filename)

                with open(pdf_path, 'wb') as file:
                    file.write(pdf_bytes)

                image.close()
                file.close()