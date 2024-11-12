import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import numpy as np

# 设置支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 (SimHei)
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 写入数据：XXX的体重和体脂数据
data = {
    "Date": [
        "2023/12/28", "2024/1/4", "2024/1/11", "2024/1/18", "2024/2/1",
        "2024/2/16", "2024/3/8", "2024/3/28", "2024/4/18", "2024/4/25",
        "2024/5/2", "2024/5/9", "2024/6/6", "2024/6/13", "2024/6/21",
        "2024/7/11", "2024/7/19", "2024/8/1", "2024/8/8", "2024/8/22",
        "2024/8/29", "2024/9/5"
    ],
    "Weight": [
        73.5, 74.8, 74.2, 74.1, 75.1, 76.2, 75.6, 73.7, 73.1, 73.0,
        73.8, 73.9, 73.0, 73.5, 74.3, 74.3, 74.0, 74.6, 75.5, 73.8,
        73.3, 73.6
    ],
    "Body Fat": [
        8.22, 8.67, 8.49, 8.43, 8.79, 9.14, 8.94, 8.29, 8.09, 8.05,
        8.33, 8.36, 8.05, 8.22, 8.5, 8.5, 8.39, 8.6, 8.9, 8.33,
        8.16, 8.26
    ]
}


# 创建 Pandas DataFrame
df = pd.DataFrame(data)

# 将日期列转换为日期格式
current_year = pd.Timestamp.now().year  # 获取当前年份
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y/%m/%d')
    except ValueError:
        return pd.to_datetime(f'{current_year}/{date_str}', format='%Y/%m/%d')

df['Date'] = df['Date'].apply(parse_date)

# 检查日期跨年的情况，如果有跨年则自动调整年份
if df['Date'].iloc[0].month == 12 and df['Date'].iloc[-1].month < 12:
    df['Date'] = df['Date'].apply(lambda x: x.replace(year=current_year - 1) if x.month == 12 else x)

# 使用插值法填补缺失的数据
df['Weight'] = df['Weight'].interpolate()
df['Body Fat'] = df['Body Fat'].interpolate()

# 创建图表，并设置图表大小
fig, ax1 = plt.subplots(figsize=(12, 6))  # 设置图表的大小为 12 x 6 英寸

# 绘制体重曲线（左侧 y 轴）
ax1.set_xlabel('日期', fontsize=24)  # 设置 x 轴标签为 '日期'，字体大小为 14
ax1.set_ylabel('体重 (kg)', fontsize=30, color='navy')  # 设置 y 轴标签，字体大小为 14，颜色为 'navy'
ax1.plot(
    df['Date'], df['Weight'],
    marker='o', markersize=8,  # 设置数据点的形状为 'o'，大小为 8
    linestyle='-', linewidth=3,  # 设置线条样式为实线，线条宽度为 1.5
    color='navy', label='体重'  # 设置线条颜色为 'navy'，标签为 '体重'
)
ax1.tick_params(axis='y', labelsize=30, labelcolor='navy')  # 设置 y 轴刻度标签的字体大小为 12，颜色为 'navy'

# 设置 x 轴刻度为每月，使用固定的月份间隔
start_date = df['Date'].min().replace(day=1)
end_date = df['Date'].max().replace(day=1) + pd.DateOffset(months=1)
dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # 每月的开始作为刻度
ax1.set_xticks(dates)

# 自定义 x 轴标签，仅在年份第一次出现时显示年份
labels = []
last_year = None
for date in dates:
    if date.year != last_year:
        labels.append(f'{date.year}年{date.month}月')
        last_year = date.year
    else:
        labels.append(f'{date.month}月')
ax1.set_xticklabels(labels, rotation=0, ha='center', fontsize=24, color='black')  # 设置 x 轴刻度标签的字体大小和颜色

# 绘制体脂率曲线（右侧 y 轴）
ax2 = ax1.twinx()  # 创建共享 x 轴的新 y 轴
ax2.set_ylabel('体脂率 (%)', fontsize=30, color='darkred')  # 设置 y 轴标签，字体大小为 14，颜色为 'darkred'
ax2.plot(
    df['Date'], df['Body Fat'],
    marker='s', markersize=8,  # 设置数据点的形状为 's'（方块），大小为 6
    linestyle='-', linewidth=3,  # 设置线条样式为实线，线条宽度为 1.5
    color='darkred', label='体脂率'  # 设置线条颜色为 'darkred'，标签为 '体脂率'
)
ax2.tick_params(axis='y', labelsize=30, labelcolor='darkred')  # 设置 y 轴刻度标签的字体大小为 12，颜色为 'darkred'

# 添加图例，并设置字体大小
ax1.legend(loc='upper left', fontsize=24)  # 在左上角添加体重曲线的图例，字体大小为 12
ax2.legend(loc='upper right', fontsize=24)  # 在右上角添加体脂率曲线的图例，字体大小为 12

# 设置标题，并调整布局
plt.title('李四体成分趋势', fontsize=20)  # 设置图表标题，字体大小为 16
fig.tight_layout()  # 自动调整子图参数，使得图表更加紧凑

# 显示图表
plt.show()
