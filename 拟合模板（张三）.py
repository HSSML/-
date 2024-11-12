import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ===============================
# 第一步：准备数据
# ===============================

# 运动表现数据（日期和秒数）
performance_data = pd.DataFrame({
    'Date': pd.to_datetime([
        '2024/4/10', '2024/4/14', '2024/5/19', '2024/5/19',
        '2024/5/28', '2024/5/29', '2024/6/16', '2024/6/28',
        '2024/6/29', '2024/7/18', '2024/9/14', '2024/9/15', '2024/9/16'
    ]),
    'Performance (s)': [
        52.55, 52.42, 52.04, 51.43, 52.20, 51.01, 52.14,
        51.23, 51.21, 51.64, 51.25, 50.64, 50.93
    ]
})

# 体脂率数据（日期和百分比）
body_fat_data = pd.DataFrame({
    'Date': pd.to_datetime([
        '2023/12/28', '2024/1/4', '2024/1/11', '2024/1/18',
        '2024/2/1', '2024/2/16', '2024/3/8', '2024/3/28',
        '2024/4/18', '2024/4/25', '2024/5/2', '2024/5/9',
        '2024/6/6', '2024/6/13', '2024/6/21', '2024/7/11',
        '2024/8/1', '2024/8/8', '2024/8/22', '2024/8/29', '2024/9/5'
    ]),
    'Body Fat (%)': [
        11.92, 11.74, 11.64, 12.13, 12.17, 11.85, 12.13, 12.83,
        11.54, 11.68, 11.46, 11.46, 10.86, 10.96, 10.82, 10.71,
        11.03, 10.93, 10.96, 10.75, 10.93
    ]
})

# ===============================
# 第二步：定义二次拟合和插值函数
# ===============================

def quadratic_fit_and_interpolate(dates, values, num_points=23):
    # 将日期转换为序数（用于拟合的数值格式）
    ordinal_dates = [d.toordinal() for d in dates]

    # 执行二次多项式拟合
    coefficients = np.polyfit(ordinal_dates, values, 2)  # 二次多项式拟合
    polynomial = np.poly1d(coefficients)  # 根据拟合生成多项式对象

    # 生成均匀分布的插值点
    interpolated_dates = np.linspace(min(ordinal_dates), max(ordinal_dates), num_points)
    interpolated_values = polynomial(interpolated_dates)  # 计算插值点对应的y值

    # 将序数转换回日期格式
    interpolated_dates = [datetime.fromordinal(int(d)) for d in interpolated_dates]
    return interpolated_dates, interpolated_values

# 为运动表现和体脂率趋势生成插值数据
interp_perf_dates, interp_perf_values = quadratic_fit_and_interpolate(
    performance_data['Date'], performance_data['Performance (s)']
)
interp_fat_dates, interp_fat_values = quadratic_fit_and_interpolate(
    body_fat_data['Date'], body_fat_data['Body Fat (%)']
)

# ===============================
# 第三步：绘制数据和趋势图
# ===============================

# 创建主图，并设置共享x轴的双y轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 添加标题
plt.title('运动表现与体脂率趋势分析（张三）', fontsize=52)

# 绘制运动表现数据（绿色点）和趋势（绿色虚线）
line1, = ax1.plot(performance_data['Date'], performance_data['Performance (s)'],
                  'go', markersize=16, label='运动表现 (秒)')
line2, = ax1.plot(interp_perf_dates, interp_perf_values,
                  'g--', linewidth=8, label='运动表现趋势 (秒)')

# 设置x轴标签和左侧y轴标签
ax1.set_xlabel('日期', fontsize=30)
ax1.set_ylabel('运动表现 (秒)', color='green', fontsize=32)

# 调整左侧y轴的刻度参数
ax1.tick_params(axis='y', labelcolor='green', labelsize=30)

# 创建右侧y轴用于绘制体脂率数据
ax2 = ax1.twinx()

# 绘制体脂率数据（红色点）和趋势（红色虚线）
line3, = ax2.plot(body_fat_data['Date'], body_fat_data['Body Fat (%)'],
                  'ro', markersize=16, label='体脂率 (%)')
line4, = ax2.plot(interp_fat_dates, interp_fat_values,
                  'r--', linewidth=8, label='体脂率趋势 (%)')

# 设置右侧y轴的标签
ax2.set_ylabel('体脂率 (%)', color='red', fontsize=32)

# 调整右侧y轴的刻度参数
ax2.tick_params(axis='y', labelcolor='red', labelsize=30)


# 设置x轴刻度定位器为每月
locator = mdates.MonthLocator()
ax1.xaxis.set_major_locator(locator)

# 自定义日期格式化器，控制年份和月份的显示
class CustomDateFormatter(mdates.DateFormatter):
    def __init__(self):
        super().__init__('%Y/%m')
        self.previous_year = None

    def __call__(self, x, pos=None):
        dt = mdates.num2date(x)
        label = dt.strftime('%m')
        if self.previous_year != dt.year:
            label = dt.strftime('%Y/%m')
            self.previous_year = dt.year
        return label

# 使用自定义日期格式化器
formatter = CustomDateFormatter()
ax1.xaxis.set_major_formatter(formatter)

# 使用 plt.setp() 设置 x 轴刻度标签的字体大小和旋转角度
plt.setp(ax1.get_xticklabels(), fontsize=20)

# 将所有图例信息组合到一起
lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]

# 将图例放置在坐标轴右上角空白区域
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize=15,
           frameon=True, fancybox=True, shadow=True)

# 调整上下边距，给标题和图例腾出空间
plt.subplots_adjust(top=0.85, bottom=0.15)

# 显示图表
plt.show()
