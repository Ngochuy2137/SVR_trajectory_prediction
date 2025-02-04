import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from multiprocessing import cpu_count

from utils.dataset import load_npz
from utils.utils import cubic_speed
from python_utils.printer import Printer
from python_utils.plotter import Plotter
# from nae_static.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
# from nae_static.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader

class SVRTrainer:
    def __init__(self, object_name, data_dir, spline_type='univariate-k3-s0', save_path='model'):
        self.object_name = object_name
        self.data_dir = data_dir
        self.spline_type = spline_type
        self.save_path = save_path
        self.dms_num = 3  # x, y, z
        self.param_grid = {
            'C': np.array([2**(i-5) for i in range(20)]), 
            'gamma': np.array([2**(i-15) for i in range(20)])
        }
        self.train_threads = cpu_count() - 4
        self.printer = Printer()
        self.plotter = Plotter()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # self.nae_data_loader = NAEDataLoader() 
        # self.input_label_generator = InputLabelGenerator()
        

    def load_and_prepare_data(self, axis_idx):
        X, y = [], []
        traj_list = os.listdir(self.data_dir)
        count = 0

        for i, traj_npz_name in enumerate(traj_list):
            if (i + 1) % 1 == 0:
                traj_data = load_npz(self.data_dir, traj_npz_name)
                xva = cubic_speed(traj_data, self.spline_type)
                print('xva shape:', xva.shape)  # (3, 3, 115)

                count += xva.shape[2]
                for j in range(xva.shape[2]):   # on steps
                    # xva[0:2, :, j].reshape(2 * self.dms_num) là vector 1 chiều với shape (6,) [x, y, z, vx, vy, vz]
                    #       reshape Chuyển từ shape (2, 3) thành một vector 1 chiều với shape (6,) [x, y, z, vx, vy, vz]
                    X.append(xva[0:2, :, j].reshape(2 * self.dms_num))  
                    y.append(xva[2, axis_idx, j])   # xva[2, axis_idx, j] là 1 scalar (gia tốc theo trục x, y hoặc z)

                    '''
                    input data là [x, y, z, vx, vy, vz] của time step j.
                    output data là gia tốc theo trục x, y hoặc z của time step j.
                    '''

        self.printer.print_green(f'Count: {count}')
        self.printer.print_green(f'X.shape: {np.array(X).shape}, y.shape: {np.array(y).shape}')
        return np.array(X), np.array(y)

    # def load_and_prepare_data(self, folder_dir, data_train_lim=None, data_val_lim=None):

    #     data_train, data_val, _ = self.nae_data_loader.load_train_val_test_dataset(folder_dir, file_format='csv')
    #     # limit data
    #     if data_train_lim is not None:
    #         data_train = data_train[:data_train_lim]
    #     if data_val_lim is not None:
    #         data_val = data_val[:data_val_lim]

    #     # 1.2 generate input and label sequences
    #     for d_train in data_train:
    #         d_train  = self.input_label_generator.preprocess_data(d_train)
    #     for d_val in data_val:
    #         d_val    = self.input_label_generator.preprocess_data(d_val)


    def train_each_axis(self, axis_idx):
        X, y = self.load_and_prepare_data(axis_idx)

        print('X shape:', X.shape)  # (11270, 6)
        print('y shape:', y.shape)  # (11270,)
        input()
        '''
        1. Khởi tạo mô hình SVR và Grid Search để tìm ra các tham số tốt nhất cho mô hình SVR: C và gamma
        '''
        model = SVR(kernel='rbf')
        grid_search = GridSearchCV(model, self.param_grid, n_jobs=self.train_threads, verbose=1)
        grid_search.fit(X, y)

        '''
        2. Huấn luyện lại mô hình SVR với các tham số tối ưu trên toàn bộ tập dữ liệu để đạt hiệu suất tốt nhất.
        '''
        best_params = grid_search.best_estimator_.get_params()
        for param, val in best_params.items():
            print(f'{param}: {val}')

        '''
        3. Huấn luyện mô hình cuối cùng với tham số tối ưu
        '''
        model = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
        model.fit(X, y)

        '''
        4. Lưu mô hình và các tham số tối ưu
        '''
        model_path = f"{self.save_path}/{self.object_name}_{axis_idx}.pkl"
        joblib.dump(model, model_path)
        with open(f"{self.save_path}/params.txt", "a") as param_f:
            param_f.write(f"{self.object_name}_{axis_idx}: C:{best_params['C']} gamma:{best_params['gamma']}\n")
        self.printer.print_green(f"Model {self.object_name} with axis ID {axis_idx} saved to {model_path}")

    def train_axisxyz(self):
        # train x, y, z
        for axis_idx in range(self.dms_num):
            print('Traing AXIS IDX:', axis_idx)
            self.train_each_axis(axis_idx)


if __name__ == '__main__':
    OBJECT_NAME = "trimmed_Bottle_115"
    DATA_DIR = f'/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/trajectory-prediction-SVR/data/{OBJECT_NAME}/training_data'

    trainer = SVRTrainer(object_name=OBJECT_NAME, data_dir=DATA_DIR)
    trainer.train_axisxyz()
