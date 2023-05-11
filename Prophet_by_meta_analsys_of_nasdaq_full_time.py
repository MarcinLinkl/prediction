import pandas as pd
from prophet import Prophet
import yfinance as yf
import plotly.graph_objects as go
import sys
import numpy as np
import matplotlib.pyplot as plt 

Ticker ='^ndx'
df = yf.download(Ticker, interval="1d")["Close"]
print(df.isnull().sum())
df.fillna(method='ffill', inplace=True)

df = df.reset_index()



df=df.rename(columns={"Date": "ds", "Close": "y"})
df=df[df['ds'].dt.dayofweek<5]

ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel(Ticker)
ax.set_xlabel('Date')

plt.show()
# REMOVING THE TIMEZONE INFORMATION
# propability to 95 (default 80)



df['ds'] = df['ds'].dt.tz_localize(None)


print(df)
m = Prophet()

# m.add_country_holidays(country_name='US')
m.fit(df)

future = m.make_future_dataframe(periods=365)	

print(future.tail())


# Python
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
)

# Python
fig1 = m.plot(forecast,uncertainty=True)

# Python
fig2 = m.plot_components(forecast)
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)
# Python
plot_components_plotly(m, forecast)
plt.show() 