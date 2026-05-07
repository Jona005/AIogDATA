import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv("DailyDelhiClimateTrain.csv")

df['meantemp_ma7'] = df['meantemp'].rolling(window=15).mean()
df['humidity_ma7'] = df['humidity'].rolling(window=15).mean()
df['wind_speed_ma7'] = df['wind_speed'].rolling(window=15).mean()
df['meanpressure_ma7'] = df['meanpressure'].rolling(window=15).mean()



fig = make_subplots(rows=2, cols=2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4", "Plot 5"))

# Mean temperature + MA
fig.add_trace(px.line(df, x='date', y='meantemp_ma7').data[0], row=1, col=1)

# Humidity + MA
fig.add_trace(px.line(df, x='date', y='humidity_ma7').data[0], row=1, col=2)

# Wind speed + MA
fig.add_trace(px.line(df, x='date', y='wind_speed_ma7').data[0], row=2, col=1)

# Pressure + MA
fig.add_trace(px.line(df, x='date', y='meanpressure_ma7').data[0], row=2, col=2)


fig.update_yaxes(range=[980, 1040], row=2, col=2)
fig.update_yaxes(range=[0, 40], row=2, col=1)

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Mean Temperature (°C)", row=1, col=1)

fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Humidity (%)", row=1, col=2)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Wind Speed (km/h)", row=2, col=1)

fig.update_xaxes(title_text="Date", row=2, col=2)
fig.update_yaxes(title_text="Mean Pressure (mbar)", row=2, col=2)

fig.show()






