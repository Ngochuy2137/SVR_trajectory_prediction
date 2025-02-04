import numpy as np
import joblib
import os
import time
import math
from config import cfg
from utils.dataset import load_npz
from utils.utils import cubic_speed, intergral_x, err_norm, plt_show, plt_show_plotly
from tqdm import tqdm

from python_utils.printer import Printer
from python_utils.plotter import Plotter
from nae_static.utils.submodules.training_utils.data_loader import DataLoader as NAEDataLoader
from collections import defaultdict

global_printer = Printer()
global_plotter = Plotter()
# from sklearn.svm import SVR




# OBJECT_NAME = 'trimmed_Bottle_115'          # "trimmed_Bottle_115"
# DATA_DIR = f'/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/trajectory-prediction-SVR/data/{OBJECT_NAME}/testing_data'
# MODEL_DIR = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/trajectory-prediction-SVR/model/trimmed_Bottle_115'

DATA_DIR = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/chip_star/3-data_preprocessed/chip_star-BW-cof-15'
OBJECT_NAME = 'chip_star-BW-cof-15'
MODEL_DIR = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/trajectory-prediction-SVR/model/chip_star-BW-cof-15-%2'


# SPLINE_TYPE = 'cubic'
SPLINE_TYPE = 'univariate-k3-s0'
'''
The univariate with k=3, s=0 is indeed a cubic spline.
'''


# test_list = os.listdir(DATA_DIR)
# if '.keep' in test_list:
#     test_list.remove(".keep")

bee = cfg['init_start']   # 0
F = cfg['frame_rate']   # 120
verbose = False

def eval_position(traj, t_now=None, show_plt=False):
    global_printer.print_green(f'      {OBJECT_NAME} - {t_now}')
    if t_now == None:
        t_now = cfg['init_time']
    t_start = time.time()                           # Ghi lại thời gian bắt đầu để đo thời gian thực thi

    pos_raw_seq = traj
    '''
    Nội suy vận tốc và gia tốc từ dữ liệu vị trí ban đầu
    posvelacc_raw có shape là (3, 3, time_steps)
    '''
    posvelacc_raw = cubic_speed(pos_raw_seq[:,bee:bee+t_now], SPLINE_TYPE)     # Tính toán vận tốc và gia tốc từ dữ liệu vị trí ban đầu (chỉ lấy khung từ bee đến bee+t_now)
    vel_raw_seq = posvelacc_raw[1].T   # shape (3, time_steps)
    acc_raw_seq = posvelacc_raw[2].T   # shape (3, time_steps)

    '''
    Sao chép dữ liệu gốc để chuẩn bị cho dữ liệu dự đoán
    shape của pos_raw_seq là (time_steps, 3)
    '''
    # pos_pred_seq = pos_raw_seq.copy()      # shape (time_steps, 3)
    pos_pred_seq = []
    acc_pred_seq = acc_raw_seq.copy()    # shape (time_steps, 3)
    
    '''
    Tạo dữ liệu đầu vào ban đầu cho mô hình
    Lấy vị trí và vận tốc tại thời điểm cuối cùng trong khoảng t_now
    - posvelacc_raw[0,:,t_now-1]: vị trí x, y, z tại thời điểm cuối cùng trong khoảng t_now
    - posvelacc_raw[1,:,t_now-1]: vận tốc x, y, z tại thời điểm cuối cùng trong khoảng t_now
    '''
    # cur_data_point = np.array([posvelacc_raw[0,:,t_now-1], posvelacc_raw[1,:,t_now-1]]).flatten()   # cur_data_point: (6,)
    cur_data_point = np.array([pos_raw_seq[t_now-1], vel_raw_seq[t_now-1]]).flatten()   # cur_data_point: (6,)      # CHANGED HERE

    traj_long = pos_raw_seq.shape[0]   # = time_steps

    '''
    Dự đoán từng khung hình tiếp theo cho đến khi hết dữ liệu
    '''
    pred_accs = []
    for j in range(traj_long - t_now - bee):
        pos_pred = [0]*3
        vel_pred = [0]*3
        acc_pred = [0]*3
        '''
        Dự đoán cho từng chiều không gian (x, y, z)
        '''
        for i in range(3): # x,y,z    
            # Tải model đã được train cho từng chiều x,y,z
            model = joblib.load(f"{MODEL_DIR}/{OBJECT_NAME}_{i}.pkl")
            acc_pred[i] = model.predict(cur_data_point.reshape(1,-1))
            # Tính toán vị trí và vận tốc mới dựa trên gia tốc dự đoán
            pos_pred[i], vel_pred[i] = intergral_x(cur_data_point[i], cur_data_point[i+3], acc_pred[i], 1/F)   # x, v, a, DT
        cur_data_point = np.array([pos_pred, vel_pred]).flatten()
        
        '''
        in the final loop, the index of acc_pred_seq and pos_pred_seq will be bee+t_now+j = traj_long - t_now - bee -1 + bee+t_now = traj_long - 1
        '''
        # pos_pred shape is (3, 1), so we need to append the first element to the list with shape (3,) before appending to pos_pred_seq
        # pos_pred_seq[bee + t_now + j] = [pre_pos_i[0] for pre_pos_i in pos_pred]
        pos_pred_seq.append([pre_pos_i[0] for pre_pos_i in pos_pred])
        # posvelacc_pred[:,:,bee+t_now+j] = np.array([pos_pred, vel_pred, acc_pred]).reshape(3,3)  # shape (3, 3, time_steps)

        acc_pred_seq[bee + t_now + j] = [pre_acc_i[0] for pre_acc_i in acc_pred]      # CHANGE HERE
        pred_accs.append(np.array(acc_pred).flatten())


        # ## find difference indices between pos_pred_seq and pos_raw_seq
        # indices_diff = np.where(acc_pred_seq != acc_raw_seq)
        # unique_indices = np.unique(indices_diff[0])
        # print(f"Các chỉ số đã thay đổi: {unique_indices}/{traj_long-1}"); input()
    
    # pred_accs = np.array(pred_accs)
    # print('pred_accs: ', pred_accs.shape); input()

    if verbose:
        print("cost time:", time.time()-t_start)
    
    # pos_raw_seq[0][1][0][-1], pos_pred_seq[0][-1]
    # print("pos",np.array(pos_raw_seq)[-1], np.array(pos_pred_seq)[-1],np.array(pos_raw_seq)[-1]-np.array(pos_pred_seq)[-1], np.linalg.norm(np.array(pos_raw_seq)[-1]-np.array(pos_pred_seq)[-1]) )
    # impact point error
    ie = err_norm(np.array(pos_raw_seq)[-1], np.array(pos_pred_seq)[-1])
    # accumulated error
    ae = err_norm(acc_raw_seq[-1], acc_pred_seq[-1])
    # future prediction length
    future_length = len(pos_pred_seq)
    
    if show_plt:
        # PLOT pos
        # plt_show_plotly([pos_raw_seq, pos_pred_seq], num=2, color=['red', 'green'], rotate_data_whose_y_up=True)
        # convert to numpy array
        pos_raw_seq = np.array(pos_raw_seq)
        pos_pred_seq = np.array(pos_pred_seq)

        # if len(pos_pred_seq)%10==0:
        #     plt_show_plotly(ground_truth=pos_raw_seq, prediction=pos_pred_seq, color_gt='red', color_pred='blue', size=5, rotate_data_whose_y_up=True)
        #     input()
        # # input(str(len(pos_pred_seq)))

    
       
    return ie, ae, future_length


def process_prediction_result(thrown_object, ie_result):
        # calculate mean and std of ie for each future prediction length
    result_plot = []
    for future_prediction_len, ie_list in ie_result.items():
        mean_ie = np.mean(ie_list)
        std_ie = np.std(ie_list)
        samp_num = len(ie_list)
        r_plot = {
            'future_prediction_len (NOT free running)': future_prediction_len, 
            'mean_ie': mean_ie, 
            'std_ie': std_ie,
            'samp_num': samp_num
        }
        result_plot.append(r_plot)
    # sort descending by future_prediction_len
    result_plot = sorted(result_plot, key=lambda x: x['future_prediction_len (NOT free running)'], reverse=True)

    x_plot = [rf['future_prediction_len (NOT free running)'] for rf in result_plot]
    y_mean_plot = [rf['mean_ie'] for rf in result_plot]
    y_std_plot = [rf['std_ie'] for rf in result_plot]
    label_x = 'Future prediction length (NOT free running)'
    label_y = 'Mean IE'
    title = f'{thrown_object} \n Mean IE vs Future prediction length (free running)'
    global_plotter.plot_bar_chart(x_values = x_plot, y_values = [y_mean_plot], y_stds=[y_std_plot], 
                                            # x_tick_distance=5, 
                                            # y_tick_distance=0.01,
                                            font_size_title=30,
                                            font_size_label=20,
                                            font_size_tick=20,
                                            font_size_bar_val=22,
                                            title=title, 
                                            x_label=label_x, 
                                            y_label=label_y,
                                            legends=None,
                                            # keep_source_order=True,
                                            bar_width=0.3,
                                            y_lim=[0, 1])
    
if __name__ == '__main__':
    res = open(f"{OBJECT_NAME}_results.txt", 'a')



    FUTURE_PREDICT_LEN_PLOT = [50, 40, 30, 20, 10]
    prediction_results = defaultdict(list)
    nae_data_loader = NAEDataLoader()
    data_train, data_val, data_test = nae_data_loader.load_train_val_test_dataset(DATA_DIR)
    data_test_pos = []
    for traj in data_test:
        traj_pos = traj['preprocess']['model_data'][:, :3]
        data_test_pos.append(traj_pos)

    
    ach_pre_t = []
    leading_time_list = []
    for idx, traj in enumerate(data_test_pos):
        impact_err = 9999
        traj_long = len(traj)
        T = cfg['init_time']-1

        good_pos_err = False

        while(impact_err > 0.01 and T < traj_long-1):
            T += 1
            impact_err, _, future_prediction_length = eval_position(traj, T, show_plt=True)
            print('         impact_err:', impact_err)
            if future_prediction_length in FUTURE_PREDICT_LEN_PLOT:
                prediction_results[future_prediction_length].append(impact_err)
            if impact_err < 0.01:
                good_pos_err = True
                break

        global_printer.print_green(f'Trajectory {idx}: Step: {T}/{traj_long} - impact_err: {impact_err}')
        # calculate leading time
        if T < traj_long-1:
            global_printer.print_green(f"Good leading time at T={T}/{traj_long}")
        else:
            global_printer.print_red(f"BAD leading time at T={T}/{traj_long}")
            input('Press Enter to continue')
        leading_time = (traj_long - T)/120.
        leading_time_list.append(leading_time)

        ach_pre_t.append((traj_long - T)/120.)


    process_prediction_result(OBJECT_NAME, prediction_results)


        # # For PLOT
        # if good_pos_err:
        #     global_printer.print_green(f"CHECKING Good position error at T={T}/{traj_long}")
        #     impact_err, _, traj_long = eval_position(traj_name, T, show_plt=True)
        #     input()
        # answer = input('Do you want to continue with a new trajectory [y/n] ?')
        # if answer == 'n':
        #     break


    
    # ach_pre_t = np.array(ach_pre_t)
    # leading_time_list = np.array(leading_time_list)
    # leading_time_mean = leading_time_list.mean()

    # res.write(f"2st LOOP: \n")
    # res.write(f"    {OBJECT_NAME}_t:\n")
    # res.write(f"    MAX:    {ach_pre_t.max()}\n")
    # res.write(f"    MIN:    {ach_pre_t.min()}\n")
    # res.write(f"    MEAN:   {ach_pre_t.mean()}\n")
    # res.write(f"    LEADING_TIME_MEAN:   {leading_time_mean}\n")
    # res.close()
        