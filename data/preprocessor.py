import os
import re
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logger import logger

class Preprocessor():
    def __init__(self, stride=9, frequency=200, window_length=1, offset=1):
        self.offset = offset
        self.stride = stride
        self.frequency = frequency
        self.length = window_length
        self.f_len = stride*3
        self.train_X, self.test_X = np.empty((0,self.f_len)), np.empty((0,self.f_len))
        self.train_Y, self.test_Y = np.empty((0,)), np.empty((0,))

    def process_folder(self, base_path, out_path):
        out_path = self.append_options(out_path)
        
        for dirpath, _, files in os.walk(base_path):
            for f in files:
                if f.startswith('.'):
                    continue
                    
                src = os.path.join(dirpath, f)
                relative_path = os.path.relpath(dirpath, base_path)
                
                if relative_path == ".":
                    target_dir = out_path
                else:
                    target_dir = os.path.join(out_path, relative_path)
                
                os.makedirs(target_dir, exist_ok=True)

                file_name_without_ext = os.path.splitext(f)[0]
                outfile = os.path.join(target_dir, file_name_without_ext)
                
                logger.info(f'>>> Extracting to: {os.path.join(relative_path, file_name_without_ext)}.npz')

                data = self.load_file(src)
                X, Y = self.process_data(data)
                self.save_processed_file(X, Y, outfile)

    def append_options(self, outpath):
        outpath += f'_s{self.stride}'
        outpath += f'_f{self.frequency}'
        outpath += f'_w{self.length}'
        outpath += f'_o{self.offset}'
        return outpath

    def load_file(self, file_path):
        data = pd.read_csv(file_path, sep='\t')
        if 'Filename' in data.columns:
            return data.drop(['Filename'], axis=1)
        return data

    def save_processed_file(self, X, Y, file_path):
        np.savez(file_path, X=X, Y=Y)

    def process_data(self, data):
        strides = [2**val for val in range(self.stride)]
        fac = (self.frequency * self.length)/strides[-1]
        window = [int(np.ceil(i*fac)) for i in strides]
        latency = 1000/self.frequency
        x = data['X_coord'].to_numpy()
        y = data['Y_coord'].to_numpy()
        c = data['Confidence'].to_numpy()
        p = data['Pattern'].to_numpy()
        X,Y = self.extract_features(x,y,c,p, window, latency)
        return X, Y
        
    def extract_features(self, x, y, conf, targets, windows, latency):
        ini = int(np.ceil(self.frequency * self.length))
        num_features = 4 * len(windows) 
        tr_tensor = np.zeros((len(x)-ini, num_features))
        tgt_tensor = np.zeros(len(targets)-ini,)
        
        for i in range(ini, len(x)):
            for j in range(len(windows)):
                start_pos, end_pos = self._get_start_end(i, windows[j])
                if start_pos == end_pos:
                    continue
        
                x_seg = x[start_pos:end_pos+1]
                y_seg = y[start_pos:end_pos+1]
                diff_x = x[end_pos] - x[start_pos]
                diff_y = y[end_pos] - y[start_pos]
                ampl = math.sqrt(diff_x**2 + diff_y**2)
                time = ((end_pos - start_pos) * latency) / 1000.0
        
                tr_tensor[i-ini][j] = ampl / time
                tr_tensor[i-ini][j + len(windows)] = math.atan2(diff_y, diff_x)
                disp = np.sqrt(np.var(x_seg) + np.var(y_seg))
                tr_tensor[i-ini][j + 2 * len(windows)] = disp
                dist = np.sqrt((x_seg - np.mean(x_seg))**2 + (y_seg - np.mean(y_seg))**2)
                std = np.std(dist)
                tr_tensor[i-ini][j + 3 * len(windows)] = std
        
            tgt_tensor[i-ini] = self._convert_label(targets[i + self.offset])
        
        return tr_tensor, tgt_tensor
    
    def _get_start_end(self, i, step):
        end_pos = i
        start_pos = i - step
        if start_pos < 0:
            start_pos = 0
        return start_pos, end_pos

    def _convert_label(self, target):
        if target == 'F': return 0
        if target == 'S': return 1
        if target == 'P': return 2
        if target == 'B': return 3
        return 3

    def load_data(self, base_path, stride=10, selected_features=['speed', 'direction']):
        logger.info(f">>> Loading data from {base_path}...")
        X_base, Y_base = None, None
        file_count = 0

        for dirpath, _, files in os.walk(base_path):
            for f in files:
                if not f.endswith('.npz'):
                    continue
                src = os.path.join(dirpath, f)
                data = np.load(src, mmap_mode='r')
                X, Y = data['X'], data['Y']
                feature_map = {'speed': 0, 'direction': 1, 'displacement': 2, 'stddev': 3}
                idx = [i for feat in selected_features for i in range(feature_map[feat]*stride, (feature_map[feat]+1)*stride)]
                X = X[:, idx]
                
                X_base, Y_base = self._stack_data((X_base, Y_base), (X, Y))
                file_count += 1
                
        logger.info(f"Loaded {file_count} files. Total samples: {len(Y_base) if Y_base is not None else 0}")

        train_X, test_X, train_Y, test_Y = train_test_split(
            X_base, Y_base, test_size=0.2, stratify=Y_base, random_state=42
        )

        logger.info(f">>> Total: {len(Y_base)} | Train: {len(train_Y)} | Test: {len(test_Y)}")
        
        return train_X, train_Y, test_X, test_Y

    def _stack_data(self, base_data, new_data):
        X_base, Y_base = base_data
        X_new, Y_new = new_data
        if X_base is None:
            return X_new, Y_new
        return np.vstack((X_base, X_new)), np.concatenate((Y_base, Y_new))