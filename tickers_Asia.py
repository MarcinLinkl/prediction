import yfinance as yf
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
# Tickery z Azji z yahoo finanse - wszystkie aktualne 
tickers_azja=['^GSPC','^N225', '^HSI', '^AXJO', '^KS11', '^BSESN', '^STI', '^TWII', '^KLSE', '^JKSE', '^NZ50', '^NSEI', '^BSESN', '^TWII', '^TWII']
# pobieram dane bez uwzględnienia strefy czasowej
df = yf.download(tickers_azja, ignore_tz=True)["Close"]
print(df)
# wyrzucam dane ^GSPC bez danych tj notna
df=df[df['^GSPC'].notna()]
# wypisuje brakujace dane
null_counts = df.isna().sum()/df.count()[0] #print fraction of NAN rows
print("Fraction % of nulls:\n",null_counts)
# Delete rows containing either 30% or more than 30% NaN Values
perc = 40.0 # Here N is 40
min_count =  int(((100-perc)/100)*df.shape[1] + 1) # number of min Nan (0.30 * ilość kolumn)
df = df.dropna( axis=0, thresh=min_count)
null_counts = df.isna().sum()/df.count()[0]  #print fraction of NAN rows
print(f"Fraction of nulls: after deleting {perc} of Nan rows\n",null_counts)
# Uzupełniam poprzendimi wartościami
df.fillna(method='ffill', inplace=True)
null_counts = df.isna().sum()/df.count()[0]  #print fraction of NAN rows
print("Fraction after ffill methods - adding previous value\n",null_counts)
# jeśli nie ma wartości (np. na początku danych)
df=df.dropna()
null_counts = df.isna().sum()/df.count()[0]  #print fraction of NAN rows
# niepowinno być już żadnych Nan
print("Fraction after dropna()\n",null_counts)
# zapisuje dane to pliku csv
df.to_csv("tickers_asian_data_stocks.csv")