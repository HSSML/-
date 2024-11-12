import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try:
    import statsmodels.api as sm
except ImportError:
    print("Error: statsmodels is not installed. Please install it using 'pip install statsmodels'.")
    exit()

def athlete_performance_analysis(body_fat_data, performance_data):
    # 将体脂数据和成绩数据转换为DataFrame
    df_fat = pd.DataFrame(body_fat_data)
    df_perf = pd.DataFrame(performance_data)

    # 将日期转换为标准格式
    df_fat['日期'] = pd.to_datetime(df_fat['日期'], format='%Y/%m/%d')
    df_perf['日期'] = pd.to_datetime(df_perf['日期'], format='%Y/%m/%d')

    # 使用插值法补足缺失的体脂数据
    fat_interp = interp1d(
        df_fat['日期'].astype(np.int64), df_fat['体脂'], kind='linear', fill_value='extrapolate'
    )

    # 将成绩数据的日期映射到最近的体脂率
    df_perf['体脂'] = fat_interp(df_perf['日期'].astype(np.int64))

    # 准备回归分析的数据
    X = df_perf[['体脂']].values  # 自变量：体脂率
    y = df_perf['成绩(秒)'].values  # 因变量：成绩

    # 进行线性回归分析
    X_with_const = sm.add_constant(X)  # 添加常数项以适应 statsmodels 的要求
    model = sm.OLS(y, X_with_const)
    results = model.fit()

    # 获取回归系数、p值、R^2等统计信息
    coef = results.params[1]  # 体脂率的回归系数
    intercept = results.params[0]  # 截距
    p_value = results.pvalues[1]  # 体脂率的p值
    r_squared = results.rsquared

    # 计算效应量 cohen's f^2
    f_squared = r_squared / (1 - r_squared) if r_squared < 1 else np.inf

    # 打印回归结果
    print("回归系数:", coef)
    print("截距:", intercept)
    print(f"p值: {p_value:.4f}")
    print(f"Cohen\'s f-squared: {f_squared:.4f}")

    # 设置中文字体（解决字体缺失问题）
    plt.rcParams['font.family'] = ['SimHei', 'Arial']  # 指定多种字体，SimHei为中文，Arial为数学符号
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常

    # 绘制回归分析图
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 绘制散点图
    plt.scatter(X, y, label='实际数据', s=300, color='teal', alpha=0.7)  # 设置颜色和透明度

    # 绘制回归线
    y_pred = results.predict(X_with_const)
    plt.plot(X, y_pred, color='darkviolet', linestyle='--', linewidth=10, label='回归线')  # 设置线条颜色、样式和宽度

    # 添加回归方程到图表中
    equation_text = f'方程: y = {coef:.3f} * x + {intercept:.3f}\nCohen\'s f-squared = {f_squared:.4f}\np值 = {p_value:.4f}'
    plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes, fontsize=25,
             bbox=dict(facecolor='white', alpha=0.5))  # 设置方程文本位置、字体大小、背景颜色和透明度

    # 标注轴和标题
    plt.xlabel('体脂率 (%)', fontsize=35, color='black')  # 设置x轴标签的字体大小和颜色
    plt.ylabel('成绩 (秒)', fontsize=35, color='black')  # 设置y轴标签的字体大小和颜色
    plt.title('体脂率与成绩之间的回归分析（李四）', fontsize=52, color='darkblue')  # 设置标题的字体大小和颜色

    # 设置刻度参数
    plt.xticks(fontsize=35, color='black')  # 设置x轴刻度字体大小和颜色
    plt.yticks(fontsize=35, color='black')  # 设置y轴刻度字体大小和颜色

    # 添加网格线
    plt.grid(True, linestyle=':', color='gray', linewidth=1, alpha=0.7)  # 设置网格线的样式、颜色、宽度和透明度

    # 添加图例
    plt.legend(fontsize=25, loc='upper right')  # 设置图例字体大小和位置

    # 显示图表
    plt.show()

# 示例数据（李四）
body_fat_data = {
    '日期': ['2024/3/28', '2024/4/18', '2024/4/25', '2024/5/2', '2024/5/9',
           '2024/6/6', '2024/6/21', '2024/7/11', '2024/7/19', '2024/8/1',
           '2024/8/8', '2024/8/22', '2024/8/29', '2024/9/5'],
    '体脂': [20.16, 19.7, 19.58, 20.06, 20.0, 19.93, 20.37, 20.09, 20.49,
           19.65, 19.61, 19.45, 19.45, 19.53]
}

performance_data = {
    '日期': ['2024/4/10', '2024/4/10', '2024/4/14', '2024/4/14', '2024/6/28',
           '2024/9/15', '2024/9/16'],
    '成绩(秒)': [59.23, 59.01, 59.99, 59.09, 59.23, 58.45, 57.76]
}

# 运行回归分析
athlete_performance_analysis(body_fat_data, performance_data)
