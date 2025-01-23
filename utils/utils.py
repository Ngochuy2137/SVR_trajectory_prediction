import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.misc import derivative
import sys
import plotly.graph_objects as go
import re

sys.path.append("..")
from config import cfg

# cubic spine for one traj
# return ((x,v,a)，3，traj.size())
# input: raw_data: (traj.size(), 3)
def cubic_speed(raw_data, spline_type='univariate-k3-sNone'):
    Xs = []
    Vs = []
    ACCs = []

    # xét từng chiều dữ liệu x, y, z
    for i in range(3):
        x, t = timestamp(raw_data[:,i])
        if spline_type == 'cubic':
            raw_f = CubicSpline(t, x)
            vec_f = raw_f.derivative(1)
            acc_f = raw_f.derivative(2)
        else:
            match = re.search(r'univariate-k(?P<k>\d+)-s(?P<s>\w+)', spline_type)
            if match:
                s_value = None if match.group('s') == 'None' else float(match.group('s'))
                k_value = int(match.group('k'))
                print(f"Found 'univariate'. s: {s_value}, k: {k_value}")
                raw_f = UnivariateSpline(t,x, k=k_value, s=s_value)
                vec_f = raw_f.derivative(n=1)
                acc_f = raw_f.derivative(n=2)
            else:
                print("The string does not contain 'univariate'.")
                raise ValueError("Invalid spline type.")
        X = []
        V = []
        ACC = []
        for j in range(t.shape[0]):
            X.append(x[j])
            V.append(vec_f(t[j]))
            ACC.append(acc_f(t[j]))
        Xs.append(X)    # có shape là (3, time_steps)
        Vs.append(V)
        ACCs.append(ACC)
    return np.array([Xs, Vs, ACCs])     # có shape là (3, 3, time_steps)


def timestamp(x):
    t = []
    for i in range(x.shape[0]):
        t.append(i/cfg['frame_rate'])
    t = np.array(t)
    return x, t


# calculate x(t+1) v(t+1)
def intergral_x(x, v, a, DT):
    x = x + v * DT + a * DT * DT / 2
    v = v + a * DT
    return x, v


def err_cal(xTrue, xPre):
    err_rate = 0.0
    for i in range(3):
        err_rate = err_rate + abs(xPre[i] - xTrue[i]) / abs(xTrue[i])
    return err_rate / 3.0 * 100.0


def err_norm(xTrue, xPre):
    return np.linalg.norm(xTrue - xPre)


def plt_show(data, num=1, color=['red']):
    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    ax.set_zlabel('Z')  # 设置z坐标轴
    for i in range(num):
        ax.scatter(data[i][:, 0], data[i][:, 1], data[i][:, 2], color=color[i], s=1)
        # ax.plot(data[i][0], data[i][1], data[i][2], color=color[i])
    plt.show()

# def plt_show_plotly(data, num=1, color=['red'], size=5, rotate_data_whose_y_up=False, title='3D Image Show'):
#     """
#     Hiển thị dữ liệu 3D sử dụng plotly với kích thước ký tự điều chỉnh và tỷ lệ trục bằng nhau,
#     thêm chữ 'end' tại điểm cuối cùng của mỗi tập dữ liệu.
    
#     Args:
#         data: Danh sách chứa các mảng dữ liệu 3D, mỗi mảng có dạng (n, 3).
#         num: Số lượng tập dữ liệu cần hiển thị.
#         color: Danh sách màu sắc cho các tập dữ liệu.
#         size: Kích thước của các ký tự tròn (mặc định là 5).
#     """
#     fig = go.Figure()
    
#     # convert all data to numpy array
#     data = [np.array(d_i) for d_i in data]
#     for i in range(num):
#         # Nếu cần xoay dữ liệu
#         if rotate_data_whose_y_up:
#             temp = data[i][:, 1].copy()  # Sao chép cột thứ 2
#             data[i][:, 1] = - data[i][:, 2]  # Gán cột thứ 3 vào cột thứ 2
#             data[i][:, 2] = temp  # Gán giá trị sao chép vào cột thứ 3
        
#         # Thêm dữ liệu chính
#         fig.add_trace(go.Scatter3d(
#             x=data[i][:, 0],
#             y=data[i][:, 1],
#             z=data[i][:, 2],
#             mode='markers',
#             marker=dict(
#                 size=size,  # Kích thước ký tự
#                 color=color[i],  # Màu sắc
#                 opacity=0.8
#             ),
#             name=f'Data {i+1}'
#         ))
        
#         # Thêm chữ 'end' tại điểm cuối cùng
#         end_point = data[i][-1]
#         fig.add_trace(go.Scatter3d(
#             x=[end_point[0]],
#             y=[end_point[1]],
#             z=[end_point[2]],
#             mode='text',
#             text=['end'],  # Nội dung chữ hiển thị
#             textposition="top center",
#             name=f'End of Data {i+1}'
#         ))
    
#     # Cập nhật layout
#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             aspectmode='cube'  # Đảm bảo tỷ lệ trục bằng nhau
#         )
#     )
    
#     # Hiển thị
#     fig.show()

def plt_show_plotly(ground_truth, prediction, color_gt='red', color_pred='blue', size=5, rotate_data_whose_y_up=False, title='3D Trajectory Comparison'):
    """
    Hiển thị quỹ đạo 3D với ground truth và prediction, tính toán và hiển thị lỗi giữa hai điểm cuối.

    Args:
        ground_truth: Mảng numpy (n, 9) chứa quỹ đạo ground truth với các cột: x, y, z, vx, vy, vz, ax, ay, az.
        prediction: Mảng numpy (m, 9) chứa quỹ đạo prediction với các cột: x, y, z, vx, vy, vz, ax, ay, az.
        color_gt: Màu sắc cho quỹ đạo ground truth (mặc định là 'red').
        color_pred: Màu sắc cho quỹ đạo prediction (mặc định là 'blue').
        size: Kích thước các ký tự hiển thị (mặc định là 5).
        rotate_data_whose_y_up: Nếu True, xoay dữ liệu để trục Y hướng lên trên.
    """
    # Xử lý xoay dữ liệu nếu cần
    if rotate_data_whose_y_up:
        ground_truth = ground_truth.copy()
        prediction = prediction.copy()

        temp_gt = ground_truth[:, 1].copy()
        ground_truth[:, 1] = -ground_truth[:, 2]
        ground_truth[:, 2] = temp_gt

        temp_pred = prediction[:, 1].copy()
        prediction[:, 1] = -prediction[:, 2]
        prediction[:, 2] = temp_pred

    fig = go.Figure()

    # Vẽ ground truth
    fig.add_trace(go.Scatter3d(
        x=ground_truth[:, 0],
        y=ground_truth[:, 1],
        z=ground_truth[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color_gt,
            symbol='circle',
            opacity=0.8
        ),
        name='Ground Truth'
    ))

    # Vẽ prediction
    fig.add_trace(go.Scatter3d(
        x=prediction[:, 0],
        y=prediction[:, 1],
        z=prediction[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color_pred,
            symbol='cross',
            opacity=0.8
        ),
        name='Prediction'
    ))

    # Tính toán khoảng cách lỗi giữa hai điểm cuối
    gt_end = ground_truth[-1, :3]  # Lấy x, y, z của điểm cuối ground truth
    pred_end = prediction[-1, :3]  # Lấy x, y, z của điểm cuối prediction
    distance = np.linalg.norm(gt_end - pred_end)

    # Vẽ đường nối hai điểm cuối bằng đường đứt đoạn
    fig.add_trace(go.Scatter3d(
        x=[gt_end[0], pred_end[0]],
        y=[gt_end[1], pred_end[1]],
        z=[gt_end[2], pred_end[2]],
        mode='lines',
        line=dict(
            color='black',
            dash='dash'
        ),
        name='Error Line'
    ))

    # Hiển thị lỗi dưới dạng text gần đường nối
    midpoint = (gt_end + pred_end) / 2  # Tính điểm giữa để hiển thị text
    fig.add_trace(go.Scatter3d(
        x=[midpoint[0]],
        y=[midpoint[1]],
        z=[midpoint[2]],
        mode='text',
        text=[f'err={distance:.2f}'],
        textposition="top center",
        name='Error Text'
    ))

    # Cập nhật layout
    fig.update_layout(
        title={
            'text': f'3D Trajectory Comparison: Prediction Length = {len(prediction)}, Error = {distance:.2f}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=35)
        },
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        )
    )

    # Hiển thị biểu đồ
    fig.show()
