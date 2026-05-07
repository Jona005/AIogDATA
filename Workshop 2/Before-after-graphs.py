import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Before and after graphs for smoothing techniques on the DailyDelhiClimateTrain dataset

df = pd.read_csv("DailyDelhiClimateTrain.csv")
df['date'] = pd.to_datetime(df['date'])

# Rolling calculations
df['meantemp_ma15'] = df['meantemp'].rolling(window=15, min_periods=1).mean()
df['humidity_ma15'] = df['humidity'].rolling(window=15, min_periods=1).mean()
df['wind_speed_ma15'] = df['wind_speed'].rolling(window=15, min_periods=1).mean()
df['meanpressure_med15'] = df['meanpressure'].rolling(window=15, min_periods=1).median()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Mean Temp", "Humidity", "Wind Speed", "Mean Pressure")
)

# --- Mean Temp ---
fig.add_trace(go.Scatter(
    x=df['date'], y=df['meantemp'],
    name="Temp (Original)", line=dict(color='blue', width=1), opacity=0.3
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df['date'], y=df['meantemp_ma15'],
    name="Temp (MA 15)", line=dict(color='blue', width=2)
), row=1, col=1)

# --- Humidity ---
fig.add_trace(go.Scatter(
    x=df['date'], y=df['humidity'],
    name="Humidity (Original)", line=dict(color='orange', width=1), opacity=0.3
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=df['date'], y=df['humidity_ma15'],
    name="Humidity (MA 15)", line=dict(color='orange', width=2)
), row=1, col=2)

# --- Wind Speed ---
fig.add_trace(go.Scatter(
    x=df['date'], y=df['wind_speed'],
    name="Wind (Original)", line=dict(color='green', width=1), opacity=0.3
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df['date'], y=df['wind_speed_ma15'],
    name="Wind (MA 15)", line=dict(color='green', width=2)
), row=2, col=1)

# --- Pressure (median smoother) ---
fig.add_trace(go.Scatter(
    x=df['date'], y=df['meanpressure'],
    name="Pressure (Original)", line=dict(color='red', width=1), opacity=0.3
), row=2, col=2)

fig.add_trace(go.Scatter(
    x=df['date'], y=df['meanpressure_med15'],
    name="Pressure (Median 15)", line=dict(color='red', width=2)
), row=2, col=2)

# Axis settings
fig.update_yaxes(range=[980, 1040], row=2, col=2)
fig.update_yaxes(range=[0, 40], row=2, col=1)

fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Value")

fig.update_layout(
    title="Original vs Smoothed Climate Signals",
    hovermode="x unified"
)

fig.show()