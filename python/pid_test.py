# -*- coding: utf-8 -*-
"""
 @File    : pid_test.py
 @Time    : 2022/8/15 上午11:55
 @Author  : yizuotian
 @Description    :
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from simple_pid import PID


class Heater:
    def __init__(self, factor):
        self.temp = 50
        self.factor = factor

    def update(self, power, dt):
        if power > 0:
            # 加热时房间温度随变量power和时间变量dt 的变化
            self.temp += 4 * self.factor * power * dt
        # 表示房间的热量损失
        self.temp -= self.factor * dt

        return self.temp


def get_points(kp, ki, kd, seconds, interval, factor):
    heater = Heater(factor)
    temp = heater.temp
    print(temp)
    # 设置PID的三个参数，以及限制输出
    pid = PID(kp, ki, kd, setpoint=temp, sample_time=None)
    pid.output_limits = (0, None)
    xs = np.arange(0, seconds, interval)
    print(xs)
    ys = []
    targets = [100] * len(xs)
    for x in xs:
        # 变量temp在整个系统中作为输出，变量temp与理想值之差作为反馈回路中的输入，通过反馈回路调节变量power的变化。
        power = pid(temp, dt=interval)
        temp = heater.update(power, interval)
        print(power, temp)
        ys.append(temp)
        pid.setpoint = 100

    return xs, ys, targets


def main_hist():
    # 将创建的模型写进主函数
    heater = Heater()
    temp = heater.temp
    # 设置PID的三个参数，以及限制输出
    pid = PID(2, 0.05, 0.06, setpoint=temp)
    pid.output_limits = (0, None)
    # 用于设置时间参数
    start_time = time.time()
    last_time = start_time
    # 用于输出结果可视化
    setpoint, y, x = [], [], []
    # 设置系统运行时间
    while time.time() - start_time < 5:
        # 设置时间变量dt
        current_time = time.time()
        dt = (current_time - last_time)
        if dt < 0.1:
            continue

        # 变量temp在整个系统中作为输出，变量temp与理想值之差作为反馈回路中的输入，通过反馈回路调节变量power的变化。
        power = pid(temp)
        temp = heater.update(power, dt)

        # 用于输出结果可视化
        x += [current_time - start_time]
        y += [temp]
        setpoint += [pid.setpoint]
        # 用于变量temp赋初值
        if current_time - start_time > 0:
            pid.setpoint = 100

        last_time = current_time

    # 输出结果可视化
    setpoint[0] = 100
    plt.plot(x, setpoint, label='target')
    plt.plot(x, y, label='PID')
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.legend()
    plt.show()


def main():
    interval = 1e-1*1
    params = [[0.2, 0, 0, 5, interval, 5],
              [0.2, 0.02, 0, 5, interval, 5],
              [0.2, 0.02, 0.01, 5, interval, 5],
              [0.2, 0.02, 0.02, 5, interval, 5]]
    num = len(params)
    cols = 2
    rows = int(np.ceil(num / 2))
    fig, axs = plt.subplots(rows, cols)
    # fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.5, hspace=1)  # 调整子图间距
    for i, param in enumerate(params):
        xs, ys, targets = get_points(*param)
        ax = axs[i // cols][i % cols]
        ax.plot(xs, targets, label='target')
        ax.plot(xs, ys, label='PID')
        ax.set_title("kp:{},ki:{},kd:{}".format(*param[:3]))
        ax.set_xlabel('time')
        ax.set_ylabel('temperature')
        ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
