import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


df = pd.read_csv("DailyDelhiClimateTrain.csv")

df['meanpressure_ma15'] = df['meanpressure'].rolling(window=15, min_periods=1).mean()
df['meanpressure_med15'] = df['meanpressure'].rolling(window=15, min_periods=1).median()


fig = make_subplots(rows=1, cols=1, subplot_titles=("Plot 1"))

# Original data
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['meanpressure'],
    name="Original",
    line=dict(color='Grey', width=1),
    opacity=0.2
), row=1, col=1)

# Rolling mean
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['meanpressure_ma15'],
    name="Rolling Mean (15)",
    line=dict(color='red', dash='dot', width=2),
    opacity=0.7
), row=1, col=1)

# Rolling median
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['meanpressure_med15'],
    name="Rolling Median (15)",
    line=dict(color='green', dash='dash', width=2),
    opacity=1
), row=1, col=1)

fig.update_yaxes(range=[980, 1040], row=1, col=1)

fig.show()

