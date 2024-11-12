import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ===============================
# 第一步：准备数据
# ===============================

# 示例数据，您可以替换为其他运动员的数据
# 运动表现数据
performance_data = pd.DataFrame({
    'Date': pd.to_datetime([
        '2024/4/10', '2024/4/10', '2024/4/14', '2024/4/14',
        '2024/6/28', '2024/9/15', '2024/9/16'
    ]),
    'Performance (s)': [
        59.23, 59.01, 59.99, 59.09, 59.23, 58.45, 57.76
    ]
})

# 体脂率数据
body_fat_data = pd.DataFrame({
    'Date': pd.to_datetime([
        '2024/3/28', '2024/4/18', '2024/4/25', '2024/5/2',
        '2024/5/9', '2024/6/6', '2024/6/21', '2024/7/11',
        '2024/7/19', '2024/8/1', '2024/8/8', '2024/8/22',
        '2024/8/29', '2024/9/5'
    ]),
    'Body Fat (%)': [
        20.16, 19.7, 19.58, 20.06, 20, 19.93, 20.37, 20.09,
        20.49, 19.65, 19.61, 19.45, 19.45, 19.53
    ]
})

# ===============================
# 第二步：定义二次拟合和插值函数
# ===============================

def quadratic_fit_and_interpolate(dates, values, num_points=20):
    # 将日期转换为序数
    ordinal_dates = [d.toordinal() for d in dates]
    # 二次多项式拟合
    coefficients = np.polyfit(ordinal_dates, values, 2)
    polynomial = np.poly1d(coefficients)
    # 插值
    interpolated_ordinals = np.linspace(min(ordinal_dates), max(ordinal_dates), num_points)
    interpolated_values = polynomial(interpolated_ordinals)
    # 序数转换回日期
    interpolated_dates = [datetime.fromordinal(int(d)) for d in interpolated_ordinals]
    return interpolated_dates, interpolated_values, polynomial

# 生成插值数据和多项式
interp_perf_dates, interp_perf_values, perf_poly = quadratic_fit_and_interpolate(
    performance_data['Date'], performance_data['Performance (s)']
)
interp_fat_dates, interp_fat_values, fat_poly = quadratic_fit_and_interpolate(
    body_fat_data['Date'], body_fat_data['Body Fat (%)']
)

# ===============================
# 第三步：自动调整 y 轴范围并对齐趋势线的起始点
# ===============================

# 找到两个数据集的共同起始日期
common_start_date = max(performance_data['Date'].min(), body_fat_data['Date'].min())

# 将日期转换为序数
start_ordinal = common_start_date.toordinal()

# 计算趋势线在共同起始日期的值
perf_start_value = perf_poly(start_ordinal)
fat_start_value = fat_poly(start_ordinal)

# 计算左侧 y 轴的范围（无额外边距）
perf_all_values = np.concatenate([performance_data['Performance (s)'], interp_perf_values])
perf_min = perf_all_values.min()
perf_max = perf_all_values.max()
ax1_ylim = (perf_min, perf_max)

# 计算右侧 y 轴的范围，使趋势线起始点对齐
fat_all_values = np.concatenate([body_fat_data['Body Fat (%)'], interp_fat_values])
fat_min = fat_all_values.min()
fat_max = fat_all_values.max()

# 计算右侧 y 轴的下限
# 使得 (perf_start_value - perf_min) / (perf_max - perf_min) = (fat_start_value - fat_min_adj) / (fat_max - fat_min_adj)
# 解方程得到 fat_min_adj
fat_min_adj = fat_start_value - ((perf_start_value - perf_min) * (fat_max - fat_min)) / (perf_max - perf_min)

ax2_ylim = (fat_min_adj, fat_max)

# ===============================
# 第四步：绘制数据和趋势图
# ===============================

# 创建图表
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制运动表现数据
line1, = ax1.plot(performance_data['Date'], performance_data['Performance (s)'],
                  'go', markersize=16, label='运动表现 (秒)')
line2, = ax1.plot(interp_perf_dates, interp_perf_values,
                  'g--', linewidth=8, label='运动表现趋势 (秒)')

# 设置左侧 y 轴
ax1.set_xlabel('日期', fontsize=32)
ax1.set_ylabel('运动表现 (秒)', color='green', fontsize=32)
ax1.tick_params(axis='y', labelcolor='green', labelsize=30)
ax1.set_ylim(ax1_ylim)

# 创建右侧 y 轴
ax2 = ax1.twinx()

# 绘制体脂率数据
line3, = ax2.plot(body_fat_data['Date'], body_fat_data['Body Fat (%)'],
                  'ro', markersize=16, label='体脂率 (%)')
line4, = ax2.plot(interp_fat_dates, interp_fat_values,
                  'r--', linewidth=8, label='体脂率趋势 (%)')

# 设置右侧 y 轴
ax2.set_ylabel('体脂率 (%)', color='red', fontsize=32)
ax2.tick_params(axis='y', labelcolor='red', labelsize=30)
ax2.set_ylim(ax2_ylim)

# 更新右侧 y 轴刻度标签
ax2_ticks = ax2.get_yticks()
ax2.set_yticks(ax2_ticks)
ax2.set_yticklabels(['{:.2f}'.format(val) for val in ax2_ticks])

# 设置 x 轴刻度定位器和格式化器
locator = mdates.MonthLocator()
ax1.xaxis.set_major_locator(locator)

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

formatter = CustomDateFormatter()
ax1.xaxis.set_major_formatter(formatter)

# 设置 x 轴刻度标签字体大小
plt.setp(ax1.get_xticklabels(), fontsize=20)

# 合并图例
lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize=15,
           frameon=True, fancybox=True, shadow=True)

# 添加标题
plt.title('运动表现与体脂率趋势分析（李四）', fontsize=52)

# 调整布局
plt.subplots_adjust(top=0.85, bottom=0.15)

# 显示图表
plt.show()
