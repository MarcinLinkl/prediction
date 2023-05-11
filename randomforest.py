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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import stock_info as si
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# pobiermay indeksy azjatyckie i SP500 ( ^GSPC )
tk_asia=['^GSPC','^N225', '^HSI', '^AXJO', '^KS11', '^BSESN', '^STI', '^TWII', '^KLSE', '^JKSE', '^NZ50', '^NSEI', '^BSESN', '^TWII', '^TWII']

callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=40,
    verbose=4,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)
# stopuj po 40echos najmniej


def load_data(indices,fromDate):
    
    # dwonload data stocks
    
    stock_data = yf.download(indices ,interval ="1d", start=fromDate,ignore_tz=True)[["Close"]]
    stock_data.columns = stock_data.columns.droplevel(0)
    print(stock_data.tail(100))

    df=stock_data
    # Ustawienie daty jako indeks DataFrame'a
    # stock_data.index = stock_data.index.tz_localize(None).tz_localize('UTC')
    # stock_data.index = pd.to_datetime(stock_data.index).date
    


    # df = stock_data.groupby(stock_data.index).mean()

    df.to_csv("dataAzja1.csv")

    # drop all ^GSPC na
    df=df[df['^GSPC'].notna()]

    null_counts = df.isna().sum()/df.count()[0] #print fraction of NAN rows
    print("nulls records\n",null_counts)

    # Delete rows containing either 30% or more than 30% NaN Values
    perc = 30.0 # Here N is 30
    min_count =  int(((100-perc)/100)*df.shape[1] + 1) # number of min Nan (0.30 * ilość kolumn)
    df = df.dropna( axis=0, thresh=min_count) 

    null_counts = df.isna().sum()/df.count()[0]  #print fraction of NAN rowsprint("nulls records",null_counts)

    print("nulls records\n",null_counts)

    df.fillna(method='ffill', inplace=True)
    
    
    
    df.dropna()
    print(null_counts)

    df=df.pct_change()
    print(df)
    df.dropna(inplace=True)
    
    null_counts = df.isna().sum()/df.count()[0]  #print fraction of NAN rows
    df['^GSPC'] = df['^GSPC'].apply(lambda x:1 if x > 0 else -1 )

    return df

def pred_this(df,df_PREDICT=0):

    scaler = MaxAbsScaler()
    #mogę jeszcze dodać później dane z poprzedniego dnia 
    print(df)
    X=scaler.fit_transform(df.drop(['^GSPC'], axis=1))
    


    
    # przygotuj dane do uczenia maszynowego
    X = df.drop(['^GSPC'], axis=1).values
    y = df['^GSPC']

    # podziel zbiór danych na zestawy treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # przygotuj listę algorytmów uczenia maszynowego i ich hiperparametrów

    # zdefiniuj i wytrenuj klasyfikator lasu losowego
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # przetestuj klasyfikator na zestawie testowym
    y_pred = clf.predict(X_test)

    # oblicz skuteczność klasyfikatora
    accuracy = accuracy_score(y_test, y_pred)
    print("Skuteczność klasyfikatora: {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':

    df=load_data(tk_asia,"2008-01-18")
    pred_this(df)


    
    

    
    

