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

global_util_printer = Printer()
global_util_plotter = Plotter()
# from sklearn.svm import SVR




OBJECT_NAME = 'trimmed_Bottle_115'          # "trimmed_Bottle_115"
DATA_TEST_DIR = f'/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/trajectory-prediction-SVR/data/{OBJECT_NAME}/testing_data'
# MODEL_DIR = f'/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/trajectory-prediction-SVR/model/{OBJECT_NAME}'
MODEL_DIR = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_fix_dismiss_acc/trajectory-prediction-SVR/model/trimmed_Bottle_115_cubic'
# SPLINE_TYPE = 'cubic'
SPLINE_TYPE = 'univariate-k3-s0'
'''
The univariate with k=3, s=0 is indeed a cubic spline.
'''


test_list = os.listdir(DATA_TEST_DIR)
if '.keep' in test_list:
    test_list.remove(".keep")

bee = cfg['init_start']   # 0
F = cfg['frame_rate']   # 120
verbose = False

def eval_position(traj_name, t_now=None, show_plt=False):
    global_util_printer.print_green(f'      {OBJECT_NAME} - {t_now}')
    if t_now == None:
        t_now = cfg['init_time']
    t_start = time.time()                           # Ghi lại thời gian bắt đầu để đo thời gian thực thi

    pos_raw_seq = load_npz(DATA_TEST_DIR,traj_name)                  # Tải dữ liệu vị trí thực tế từ tệp .npz
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
    pos_err = err_norm(np.array(pos_raw_seq)[-1], np.array(pos_pred_seq)[-1])
    acc_err = err_norm(acc_raw_seq[-1], acc_pred_seq[-1])

    
    if show_plt:
        # PLOT pos
        plt_show_plotly([pos_raw_seq, pos_pred_seq], num=2, color=['red', 'green'], rotate_data_whose_y_up=True)
        # PLOT acc
        # pred_accs = np.array(pred_accs)
        # # swap shape from (time_steps, 3) to (3, time_steps)
        # pred_accs = pred_accs.t_now
        # time_axis = np.arange(pred_accs.shape[1])
        # global_util_plotter.plot_line_chart(x_values=time_axis, y_values=[pred_accs[0], pred_accs[1], pred_accs[2]], \
        #                                 title=f'111111 Acceleration - {OBJECT_NAME}', \
        #                                 x_label='Time step', y_label='Acceleration x', \
        #                                 legends=['acc_x', 'acc_y', 'acc_z'])
        input()
       
    return pos_err, acc_err, traj_long


if __name__ == '__main__':
    # pos_errs = []
    # acc_errs = []
    # for traj_name in tqdm(test_list):
    #     global_util_printer.print_green(f"Evaluating {traj_name}")
    #     pos_err, acc_err, _ = eval_position(traj_name)
    #     # print(f"The pos error is {pos_err}, acc error is {acc_err}")
    #     pos_errs.append(pos_err)
    #     acc_errs.append(acc_err)
    # pos_errs = np.array(pos_errs)
    # acc_errs = np.array(acc_errs)
    res = open(f"{OBJECT_NAME}_results.txt", 'a')
    # res.write(f"1st LOOP: \n")
    # res.write(f"    {OBJECT_NAME}_acc_err:\n")
    # res.write(f"    MAX:    {acc_errs.max()}\n")
    # res.write(f"    MIN:    {acc_errs.min()}\n")
    # res.write(f"    MEAN:   {acc_errs.mean()}\n")

    
    ach_pre_t = []
    leading_time_list = []
    for traj_name in tqdm(test_list):
        pos_err = 9999
        traj_long = 9999
        T = cfg['init_time']-1

        good_pos_err = False
        end_traj = False
        while(pos_err > 0.01 and T < traj_long-1):
            T += 1
            pos_err, _, traj_long = eval_position(traj_name, T, show_plt=False)
            print('         pos_err:', pos_err)
            if pos_err < 0.01:
                good_pos_err = True
                break
            if T == traj_long-1:
                end_traj = True

        global_util_printer.print_green(f'Trajectory {traj_name}: Step: {T}/{traj_long} - pos_err: {pos_err}')
        # calculate leading time
        if T < traj_long-1:
            global_util_printer.print_green(f"Good leading time at T={T}/{traj_long}")
        else:
            global_util_printer.print_red(f"BAD leading time at T={T}/{traj_long}")
            input('Press Enter to continue')
        leading_time = (traj_long - T)/120.
        leading_time_list.append(leading_time)

        ach_pre_t.append((traj_long - T)/120.)

        # # For PLOT
        # if good_pos_err:
        #     global_util_printer.print_green(f"CHECKING Good position error at T={T}/{traj_long}")
        #     pos_err, _, traj_long = eval_position(traj_name, T, show_plt=True)
        #     input()
        # answer = input('Do you want to continue with a new trajectory [y/n] ?')
        # if answer == 'n':
        #     break

    ach_pre_t = np.array(ach_pre_t)
    leading_time_list = np.array(leading_time_list)
    leading_time_mean = leading_time_list.mean()

    res.write(f"2st LOOP: \n")
    res.write(f"    {OBJECT_NAME}_t:\n")
    res.write(f"    MAX:    {ach_pre_t.max()}\n")
    res.write(f"    MIN:    {ach_pre_t.min()}\n")
    res.write(f"    MEAN:   {ach_pre_t.mean()}\n")
    res.write(f"    LEADING_TIME_MEAN:   {leading_time_mean}\n")
    res.close()
        