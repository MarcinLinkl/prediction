import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score
from datetime import date, timedelta
import tensorflow as tf
from yahoo_fin import stock_info as si
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

tickers_azja=['^GSPC','^N225', '^HSI', '^AXJO', '^KS11', '^BSESN', '^STI', '^TWII', '^KLSE', '^JKSE', '^NZ50', '^NSEI', '^BSESN', '^TWII', '^TWII']


callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=17,
    verbose=4,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)
def load_data(list_index,fromDate):
	
	# Pobranie danych z giełdy
	
	stock_data = yf.download(list_index ,interval ="1d", start=fromDate,ignore_tz=True)[["Close"]]
	stock_data.columns = stock_data.columns.droplevel(0)
	print(stock_data.tail(100))

	df=stock_data
	# # Ustawienie daty jako indeks DataFrame'a
	# stock_data.index = stock_data.index.tz_localize(None).tz_localize('UTC')
	# stock_data.index = pd.to_datetime(stock_data.index).date
	


	# df = stock_data.groupby(stock_data.index).mean()

	df.to_csv("dataAzja1.csv")

	# drop all ^GSPC na
	df=df[df['^GSPC'].notna()]

	null_counts = df.isna().sum()/df.count()[0] #print fraction of NAN rows
	print(null_counts)

	# Delete rows containing either 30% or more than 30% NaN Values
	perc = 25.0 # Here N is 30
	min_count =  int(((100-perc)/100)*df.shape[1] + 1) # number of min Nan (0.30 * ilość kolumn)
	df = df.dropna( axis=0, thresh=min_count) 

	null_counts = df.isna().sum()/df.count()[0]  #print fraction of NAN rows
	print(null_counts)
	df.fillna(method='ffill', inplace=True)
	print(df)
	
	
	df.dropna()
	print(null_counts)

	df=df.pct_change()
	df.dropna(inplace=True)
	print(df)

	
	print(df)
	null_counts = df.isna().sum()/df.count()[0]  #print fraction of NAN rows

	return df


def pred_this(df,df_PREDICT):

	df['^GSPC'] = df['^GSPC'].apply(lambda x: x > 0) #mogę jeszcze dodać później dane z poprzedniego dnia do nauki
	
	df.to_csv("dataAzja1.csv")
	# Podział danych na zestawy treningowe i testowe

	X_train, X_test, y_train, y_test = train_test_split(df.drop(['^GSPC'], axis=1), df['^GSPC'], test_size=0.2,shuffle=False)
	# Tworzenie modelu sieci neuronowej
	
	model = Sequential()
	model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
	
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.05))
	model.add(Dense(1, activation='linear'))
	
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

	# Trenowanie modelu
	model.fit(X_train, y_train, epochs=1270, batch_size=32, validation_data=(X_test, y_test))
	# Wykonanie prognozy na podstawie danych testowych

	# Pobranie 30 ostatnich rekordów z zestawu testowego
	X_last_30 = X_test.tail(30)
	print(X_last_30)
	# Prognozowanie dla 30 ostatnich rekordów
	y_pred_last_30 = model.predict(X_last_30)
	print(y_pred_last_30)
	y_pred_last_30 = [1 if y>=0.5 else 0 for y in y_pred_last_30]

	# Pobranie odpowiednich wartości z zestawu testowego
	y_real_last_30 = y_test.tail(30).values

	# Tworzenie tabeli z prognozą i rzeczywistymi danymi
	result_table = pd.DataFrame({'Predicted': y_pred_last_30, 'Real': y_real_last_30})
	result_table.columns = ['Predicted', 'Real']
	# Dodanie date index
	stock_data=df.tail(30)
	result_table['Date']=stock_data.index

	# tworzenie kolumny z dokładnością
	result_table['Accuracy'] = (result_table['Predicted'] == result_table['Real']).astype(int)

	# Wyświetlenie tabeli
	print(result_table)
	pred_value=model.predict(df_PREDICT)
	y_pred_last_day = [1 if y>=0.5 else 0 for y in pred_value]
	print(f"Tomorow index will be {y_pred_last_day}, the value is {pred_value}")
	print("Acurency last 30 days ",result_table['Accuracy'].sum()/len(result_table['Accuracy']))


def load_last_data(list_index):
	list_index.remove('^GSPC')
	today=date.today()
	day2sback = today - timedelta(days=2)
	stock_data = yf.download(list_index , start=day2sback,ignore_tz=True)[["Close"]]
	stock_data.columns = stock_data.columns.droplevel(0)
	null_counts = stock_data.isna().sum()/stock_data.count()[0] #print fraction of NAN rows
	print(null_counts)
	stock_data.fillna(method='ffill', inplace=True)
	df=stock_data.pct_change()
	print(df)
	print(df.tail(1))
	return df.tail(1)

if __name__ == '__main__':
	start_date='2020-01-18'
	df=load_data(tickers_azja,start_date)
	print(df)
	df2=load_last_data(tickers_azja)
	print(df2)
	pred_this(df,df2)
	
	

	
	

