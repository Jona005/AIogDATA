import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv("DailyDelhiClimateTrain.csv")



fig = make_subplots(rows=2, cols=2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4", "Plot 5"))

fig.add_trace(px.line(df, x = 'date', y = 'meantemp').data[0], row=1, col=1)
fig.add_trace(px.line(df, x = 'date', y = 'humidity').data[0], row=1, col=2)
fig.add_trace(px.line(df, x = 'date', y = 'wind_speed').data[0], row=2, col=1)
fig.add_trace(px.line(df, x = 'date', y = 'meanpressure').data[0], row=2, col=2)
fig.update_yaxes(range=[980, 1040], row=2, col=2)

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Mean Temperature (°C)", row=1, col=1)

fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Humidity (%)", row=1, col=2)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Wind Speed (km/h)", row=2, col=1)

fig.update_xaxes(title_text="Date", row=2, col=2)
fig.update_yaxes(title_text="Mean Pressure (mbar)", row=2, col=2)

fig.show()






