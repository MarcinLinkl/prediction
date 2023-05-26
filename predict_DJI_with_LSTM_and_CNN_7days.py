import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

# download DJI
# stock_data = yf.download("^DJI" ,interval ="1d",ignore_tz=True)[["Close"]]
# stock_data.to_csv("DJI.csv")

data = pd.read_csv('DJI.csv').set_index('Date')

data['change'] = data['Close'].pct_change()

data.dropna(inplace=True)

data['change'] = np.where(data['change']>0, 1, 0)

# Use the last 7 days to predict the next day
n_steps = 7
X_train, y_train = [], []
for i in range(n_steps, len(data['change'])):
    X_train.append(data['change'][i-n_steps:i])
    y_train.append(data['change'][i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Create the LSTM and CNN model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(64))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(1, activation='sigmoid'))
optimizer = RMSprop(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation), callbacks=[early_stopping, checkpoint])


# Evaluate the performance of the model
accuracy = model.evaluate(X_validation, y_validation, verbose=0)[1]

print("Accuracy: ", accuracy)

predict_20days=X_train[-20:]
y_pred = model.predict(predict_20days)

last20days=data.tail(20)

# print(last20days)


last20days["prediction"] = np.where(y_pred > 0.5, 1, 0)
last20days["accuracy"] = last20days["prediction"] == last20days["change"]

print(last20days)

acu=np.where(last20days["prediction"] == last20days["change"], 1, 0).mean()

print(f'accuracy is {acu}')
