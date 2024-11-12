import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# ===============================
# 第一步：准备数据
# ===============================

# 运动表现数据（日期和秒数）
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
# 第三步：绘制原始数据和拟合曲线（逐帧形式）
# ===============================

# 创建一个 Plotly 图形对象
frames = []
num_frames = len(interp_perf_dates)
for i in range(1, num_frames + 1):
    frame = go.Frame(
        data=[
            go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Performance (s)'],
                mode='markers',
                marker=dict(color='green', size=10),
                name='运动表现 (秒)',
                yaxis='y1'
            ),
            go.Scatter(
                x=body_fat_data['Date'],
                y=body_fat_data['Body Fat (%)'],
                mode='markers',
                marker=dict(color='red', size=10),
                name='体脂率 (%)',
                yaxis='y2'
            ),
            go.Scatter(
                x=interp_perf_dates[:i],
                y=interp_perf_values[:i],
                mode='lines',
                line=dict(color='green', dash='dash', width=4),
                name='运动表现拟合曲线',
                yaxis='y1'
            ),
            go.Scatter(
                x=interp_fat_dates[:i],
                y=interp_fat_values[:i],
                mode='lines',
                line=dict(color='red', dash='dash', width=4),
                name='体脂率拟合曲线',
                yaxis='y2'
            )
        ],
        name=f'frame_{i}'
    )
    frames.append(frame)

fig = go.Figure(
    data=[
        go.Scatter(
            x=performance_data['Date'],
            y=performance_data['Performance (s)'],
            mode='markers',
            marker=dict(color='green', size=10),
            name='运动表现 (秒)',
            yaxis='y1'
        ),
        go.Scatter(
            x=body_fat_data['Date'],
            y=body_fat_data['Body Fat (%)'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='体脂率 (%)',
            yaxis='y2'
        ),
        go.Scatter(
            x=interp_perf_dates[:1],
            y=interp_perf_values[:1],
            mode='lines',
            line=dict(color='green', dash='solid', width=4),
            name='运动表现拟合曲线',
            yaxis='y1'
        ),
        go.Scatter(
            x=interp_fat_dates[:1],
            y=interp_fat_values[:1],
            mode='lines',
            line=dict(color='red', dash='solid', width=4),
            name='体脂率拟合曲线',
            yaxis='y2'
        )
    ],
    layout=go.Layout(
        title='运动表现与体脂率趋势分析（逐帧形式）',
        xaxis=dict(title='日期'),
        yaxis=dict(
            title='运动表现 (秒)',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            side='left'
        ),
        yaxis2=dict(
            title='体脂率 (%)',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(
                        label='拟合',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            mode='immediate'
                        )]
                    ),
                    dict(
                        label='暂停',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method='animate',
                        args=[[f'frame_{i}'], dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=0))],
                        label=str(i)
                    ) for i in range(1, num_frames + 1)
                ],
                transition=dict(duration=0),
                x=0,
                y=-0.1,
                currentvalue=dict(font=dict(size=12), prefix='帧: ', visible=True),
                len=1.0
            )
        ]
    ),
    frames=frames
)

# ===============================
# 第四步：创建 Dash 应用和交互
# ===============================

# 创建 Dash 应用
app = dash.Dash(__name__)

# 设置应用布局
app.layout = html.Div([
    dcc.Graph(
        id='trend-graph',
        figure=fig,
        config={'scrollZoom': True, 'displayModeBar': True, 'editable': True}
    ),
    html.Div(id='selected-data')
])

# 回调函数，捕获选定的数据点
@app.callback(
    Output('selected-data', 'children'),
    [Input('trend-graph', 'selectedData')]
)
def display_selected_data(selectedData):
    if selectedData is None:
        return "请选择数据点进行查看。"
    points = selectedData['points']
    selected_info = [
        f"日期: {point['x']}, 数值: {point['y']:.2f}" for point in points
    ]
    return html.Ul([html.Li(info) for info in selected_info])

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
