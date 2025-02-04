import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from multiprocessing import cpu_count

# from utils.dataset import load_npz
# from utils.utils import cubic_speed
from python_utils.printer import Printer
from python_utils.plotter import Plotter
from nae_static.utils.submodules.training_utils.input_label_generator import InputLabelGenerator
from nae_static.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
import wandb

global_printer = Printer()
global_plotter = Plotter()

class SVRTrainer:
    def __init__(self, object_name, data_dir, spline_type='univariate-k3-s0'):
        self.object_name = object_name
        self.data_dir = data_dir
        self.spline_type = spline_type
        self.dms_num = 3  # x, y, z
        self.param_grid = {
            'C': np.array([2**(i-5) for i in range(20)]), 
            'gamma': np.array([2**(i-15) for i in range(20)])
        }
        self.train_threads = cpu_count() - 4

        self.save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model', object_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.nae_data_loader = NAEDataLoader() 
        self.input_label_generator = InputLabelGenerator()
        
    def load_and_prepare_data(self, folder_dir, data_train_lim=None, data_test_lim=None):
        inputs_train = []
        labels_train = []
        inputs_test = []
        labels_test = []
        data_train, _, data_test = self.nae_data_loader.load_train_val_test_dataset(folder_dir, file_format='csv')
        # limit data
        if data_train_lim is not None:
            data_train = data_train[:data_train_lim]
        if data_test_lim is not None:
            data_test = data_test[:data_test_lim]

        # 1.2 generate input and label sequences
        for d_train in data_train:
            xya_traj_i  = self.input_label_generator.preprocess_data(d_train, acc_norm=False)
            for point in xya_traj_i:
                if point.shape[0] != 9:
                    raise ValueError(f'point shape is not 9: {point.shape}')
                inputs_train.append(point[:6])
                labels_train.append(point[6:])

        for d_test in data_test:
            xya_traj_i    = self.input_label_generator.preprocess_data(d_test)
            for point in xya_traj_i:
                if point.shape[0] != 9:
                    raise ValueError(f'point shape is not 9: {point.shape}')
                inputs_test.append(point[:6])
                labels_test.append(point[6:])

        inputs_train = np.array(inputs_train)
        labels_train = np.array(labels_train)
        inputs_test = np.array(inputs_test)
        labels_test = np.array(labels_test)
        return (inputs_train, labels_train), (inputs_test, labels_test)


    def train_each_axis(self, axis_idx, x_train, y_train):
        # print('X shape:', x_train.shape)  # (11270, 6)
        # print('y shape:', y_train.shape)  # (11270,)
        # input()
        '''
        1. Khởi tạo mô hình SVR và Grid Search để tìm ra các tham số tốt nhất cho mô hình SVR: C và gamma
        '''
        global_printer.print_green('    Step 1: Init SVR model and Grid Search')
        model = SVR(kernel='rbf')
        grid_search = GridSearchCV(model, self.param_grid, n_jobs=self.train_threads, verbose=1)
        grid_search.fit(x_train, y_train)

        '''
        2. Huấn luyện lại mô hình SVR với các tham số tối ưu trên toàn bộ tập dữ liệu để đạt hiệu suất tốt nhất.
        '''
        global_printer.print_green('    Step 2: Find best params C and gamma')
        best_params = grid_search.best_estimator_.get_params()
        for param, val in best_params.items():
            print(f'{param}: {val}')

        '''
        3. Huấn luyện mô hình cuối cùng với tham số tối ưu
        '''
        global_printer.print_green('    Step 3: Train final model with best params')
        model = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
        model.fit(x_train, y_train)

        '''
        4. Lưu mô hình và các tham số tối ưu
        '''
        global_printer.print_green('    Step 4: Save model and best params')
        model_path = f"{self.save_path}/{self.object_name}_{axis_idx}.pkl"
        joblib.dump(model, model_path)
        with open(f"{self.save_path}/params.txt", "a") as param_f:
            param_f.write(f"{self.object_name}_{axis_idx}: C:{best_params['C']} gamma:{best_params['gamma']}\n")
        global_printer.print_green(f"       Model {self.object_name} with axis ID {axis_idx} saved to {model_path}")

    def train_axisxyz(self, data_dir, data_train_lim=None, data_test_lim=None):
        self.data_train, self.data_test = self.load_and_prepare_data(folder_dir=data_dir, data_train_lim=data_train_lim, data_test_lim=data_test_lim)
        x_train = self.data_train[0]
        y_train = self.data_train[1]
        # train x, y, z
        for axis_idx in range(self.dms_num):
            global_printer.print_blue(f'----------  Traing AXIS IDX: {axis_idx}  ----------', background=True)
            self.train_each_axis(axis_idx=axis_idx, x_train=x_train, y_train=y_train[:, axis_idx])


if __name__ == '__main__':
    DATA_DIR = f'/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/ball/3-data-augmented/data_plit'
    OBJECT_NAME = "ball"
    DATA_TRAIN_LIM = None
    DATA_TEST_LIM = None

    trainer = SVRTrainer(object_name=OBJECT_NAME, data_dir=DATA_DIR)
    trainer.train_axisxyz(data_dir=DATA_DIR, data_train_lim=DATA_TRAIN_LIM, data_test_lim=DATA_TEST_LIM)
