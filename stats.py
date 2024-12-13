import os
import json

class Stats:
    def __init__(self, run_dir):
        
        self.run_dir = run_dir
        self.stat_file = os.path.join(self.run_dir, 'stats.json')
        self.data = dict()
    
    def __call__(self, key, value):
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
    
    def save(self, path = None):
        save_path = path if path is not None else self.stat_file

        with open(save_path, 'w') as file:
            json.dump(self.data, file)
    
    def load(self, path, mode):
        with open(path, 'r') as file:
            external_data = json.load(file)
        
        if mode == 'append':
            data = {x : self.data[x] + external_data[x] for x in self.data}
            self.data = data
        elif mode == 'overwrite':
            self.data = external_data
        else:
            raise NotImplementedError()
    
    def load_saved(self):
        with open(self.stat_file, 'r') as file:
            self.data = json.load(file)