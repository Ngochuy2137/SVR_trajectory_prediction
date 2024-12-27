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

def plt_show_plotly(data, num=1, color=['red'], size=5, rotate_data_whose_y_up=False, title='3D Image Show'):
    """
    Hiển thị dữ liệu 3D sử dụng plotly với kích thước ký tự điều chỉnh và tỷ lệ trục bằng nhau,
    thêm chữ 'end' tại điểm cuối cùng của mỗi tập dữ liệu.
    
    Args:
        data: Danh sách chứa các mảng dữ liệu 3D, mỗi mảng có dạng (n, 3).
        num: Số lượng tập dữ liệu cần hiển thị.
        color: Danh sách màu sắc cho các tập dữ liệu.
        size: Kích thước của các ký tự tròn (mặc định là 5).
    """
    fig = go.Figure()
    
    # convert all data to numpy array
    data = [np.array(d_i) for d_i in data]
    for i in range(num):
        # Nếu cần xoay dữ liệu
        if rotate_data_whose_y_up:
            temp = data[i][:, 1].copy()  # Sao chép cột thứ 2
            data[i][:, 1] = - data[i][:, 2]  # Gán cột thứ 3 vào cột thứ 2
            data[i][:, 2] = temp  # Gán giá trị sao chép vào cột thứ 3
        
        # Thêm dữ liệu chính
        fig.add_trace(go.Scatter3d(
            x=data[i][:, 0],
            y=data[i][:, 1],
            z=data[i][:, 2],
            mode='markers',
            marker=dict(
                size=size,  # Kích thước ký tự
                color=color[i],  # Màu sắc
                opacity=0.8
            ),
            name=f'Data {i+1}'
        ))
        
        # Thêm chữ 'end' tại điểm cuối cùng
        end_point = data[i][-1]
        fig.add_trace(go.Scatter3d(
            x=[end_point[0]],
            y=[end_point[1]],
            z=[end_point[2]],
            mode='text',
            text=['end'],  # Nội dung chữ hiển thị
            textposition="top center",
            name=f'End of Data {i+1}'
        ))
    
    # Cập nhật layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'  # Đảm bảo tỷ lệ trục bằng nhau
        )
    )
    
    # Hiển thị
    fig.show()