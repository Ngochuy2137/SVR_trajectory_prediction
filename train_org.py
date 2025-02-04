from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from utils.dataset import load_data, load_npz
from utils.utils import cubic_speed, timestamp
import joblib
import pandas as pd
import numpy as np
import os
from multiprocessing import cpu_count

from python_utils.printer import Printer
from python_utils.plotter import Plotter

global_util_printer = Printer()
global_util_plotter = Plotter()

# Xác định lưới tham số (param_grid) để tìm kiếm giá trị tối ưu cho mô hình SVR
# Các tham số bao gồm 'C' (điều chỉnh mức độ phạt cho lỗi) và 'gamma' (độ ảnh hưởng của các điểm trong kernel RBF)
# Lựa chọn sử dụng hàm mũ lũy thừa của 2 để tăng độ chính xác trong tìm kiếm tham số
# param_grid = {'C': np.linspace(0.1,3,30), 'gamma': np.linspace(0.01,0.6,60)}
param_grid = {
    'C': np.array([2**(i-5) for i in range(20)]), 
    'gamma': np.array([2**(i-15) for i in range(20)])
}
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], 'gamma': [10, 1, 0.1, 0.01, 0.001]}

dms_num = 3 # only pos
OBJECT_NAME = "trimmed_Bottle_115"
DATA_DIR = f'/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/trajectory-prediction-SVR/data/{OBJECT_NAME}/training_data'
# SPLINE_TYPE = 'cubic'
SPLINE_TYPE = 'univariate-k3-s0'
'''
The univariate with k=3, s=0 is indeed a cubic spline.
'''


train_threads = cpu_count() - 4 # for stable train
save_path = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

'''
Huấn luyện mô hình SVR dựa trên dữ liệu vị trí và lưu mô hình.
    Args:
        dms_idx: Chỉ số của thành phần động học (x, y, hoặc z).
        object_name: Tên đối tượng (ví dụ: 'boomerang').
'''
def train_pos(dms_idx):
    # traj_dir = f"npz_aug_scp/{object_name}/train_q"
    traj_list = os.listdir(DATA_DIR)
    model = SVR(kernel='rbf')
    X = [] # zeta = {[x; v]}
    y = [] # Y = {a}
    count = 0
    for i,traj_npz_name in enumerate(traj_list):
        # choose the raw data
        # Chỉ chọn dữ liệu ở mỗi bước thứ 20 để giảm tải
        if (i+1) % 20 == 0:
            traj_data = load_npz(DATA_DIR, traj_npz_name)       # shape (time_steps, 3)
            xva = cubic_speed(traj_data, SPLINE_TYPE)                    # shape ((x, v, a), 3, time_steps)


            # # plot ax, ay, az
            # acc_x = xva[2,0,:]
            # acc_y = xva[2,1,:]
            # acc_z = xva[2,2,:]
            # # x_values = np.arange(len(acc_x))
            # time = timestamp(acc_x)[1]
            # vel_interpolation_method = 'cubic'
            # acc_interpolation_method = 'cubic'
            # global_util_plotter.plot_line_chart(x_values=time, y_values=[acc_x, acc_y, acc_z], \
            #                                     title=f'222222 Acceleration - {object_name}  |  VEL_interpolate: {vel_interpolation_method} - ACC_interpolate: {acc_interpolation_method}', \
            #                                     x_label='Time step', y_label='Acceleration x', \
            #                                     legends=['acc_x', 'acc_y', 'acc_z'])
            # input()


            count += xva.shape[2]
            for i in range(xva.shape[2]):   # shape[2] = time_steps
                X.append(xva[0:2,:,i].reshape(2*dms_num))   # Get x, v at each time step    -> Input data
                y.append(xva[2,dms_idx,i])                  # Get a at each time step       -> Label
    '''
    Input data - X: được gộp từ x, v từ tất cả quỹ đạo
    Label - y: được gộp từ a từ tất cả quỹ đạo
    - Mô hình SVR không phụ thuộc vào chuỗi thời gian, chỉ phụ thuộc vào từng điểm dữ liệu
    - Dữ liệu từ mỗi điểm dữ liệu (bước thời gian) được coi là một điểm dữ liệu độc lập.
    '''
    X = np.array(X)
    y = np.array(y)
    global_util_printer.print_green(f'count: {count}')
    global_util_printer.print_green(f'X.shape: {X.shape}, y.shape: {y.shape}')
    
    # Sử dụng GridSearchCV để tìm tham số tối ưu từ lưới tham số param_grid
    grid_search = GridSearchCV(model, param_grid, n_jobs = train_threads, verbose=1)
    grid_search.fit(X, y)
    # Lấy các tham số tối ưu từ kết quả tìm kiếm
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(X, y)

    # Lưu mô hình đã huấn luyện
    model_path = f"{save_path}/{OBJECT_NAME}_{dms_idx}.pkl"
    joblib.dump(model, model_path)
    # Ghi các tham số tối ưu vào file params.txt để tham khảo sau này
    param_f = open(f"{save_path}/params.txt", "a")
    param_f.write(f"{OBJECT_NAME}_{dms_idx}: C:{best_parameters['C']} gamma:{best_parameters['gamma']}\n")
    param_f.close()
    global_util_printer.print_green(f"Model {OBJECT_NAME}_{dms_idx} was saved to {model_path}")


if __name__ == '__main__':
    '''
    Với mỗi dữ liệu trajectory, huấn luyện mô hình SVR cho để dự đoán cho từng giá trị gia tốc theo trục x, y, z
    i = 0 -> x
    i = 1 -> y
    i = 2 -> z
    '''
    for i in range(dms_num):
        train_pos(i)
