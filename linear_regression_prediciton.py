import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import sys

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Pobierz dane historyczne cen gazu ziemnego
# tk = yf.Ticker("NG=F")
# df = tk.history(period="max")

# Dodaj dodatkowe zmienne, które mogą mieć wpływ na ceny gazu
# data = yf.download("NG=F CL=F EUR=X ^GSPC", period="max", interval="1wk")["Close"]
# first_overall = pd.to_datetime("01-01-1900")
# for column in data.columns:
# 	start_date = data[column].first_valid_index()
# 	if start_date > first_overall:
# 		first_overall = start_date
# print(f"Overall first data:{first_overall}")
# data = data.loc[first_overall:]

# nie max a od daty

df = yf.download("NG=F CL=F EUR=X ^GSPC", start = "2017-12-01", interval="1d")["Close"]
print(df.isnull().sum())
df.fillna(method='ffill', inplace=True)

# Wybierz zmienną docelową (ceny gazu ziemnego) i zmienne objaśniające
X = df.drop(columns=['NG=F'])

y = df['NG=F']
print(X)
# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Użyj regresji liniowej do wytrenowania modelu
model = LinearRegression().fit(X_train, y_train)
# Ocen model na zbiorze testowym
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print(f'Model accuracy: {score:.2f}')

# Wyświetl wykres z cenami rzeczywistymi i przewidywanymi
plt.plot(y_test, label='Real gas prices')
index_range=pd.date_range(y_test.index[-1], periods=len(predictions))
predictions=pd.DataFrame(predictions, index=index_range)

print(predictions)
plt.plot(predictions, label='Predicted gas prices')
plt.legend()
plt.show()

