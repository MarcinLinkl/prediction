import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('tickers_asian_data_stocks.csv').set_index('Date').pct_change().dropna()
# SCALING
y = df['^GSPC'].apply(lambda x: x > 0)
X=df.drop("^GSPC",axis=1)


# scaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)



# Use the last 7 days to predict the next day of Sp500 based on 11 indecies close prices
n_steps = 9
X_train, y_train= [],[]
for i in range(n_steps, len(scaled_data)):
    X_train.append(scaled_data[i-n_steps:i])
    y_train.append(y[i])
# Podział na zbiór treningowy i testowy
X_train, y_train = np.array(X_train), np.array(y_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



print(X_train.shape)
# shape is (3803, 7, 11)



X_train=X_train.reshape(X_train.shape[0], -1)
X_test=X_test.reshape(X_test.shape[0], -1)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = dt.score(X_test, y_test)
print("Accuracy of Decision Tree: ", dt_acc)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)
print("Accuracy of Random Forest: ", rf_acc)
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm_acc = svm.score(X_test, y_test)
print("Accuracy of SVM: ", svm_acc)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_acc = log_reg.score(X_test, y_test)
print("Accuracy of Logistic Regression: ", log_reg_acc)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)
print("Accuracy of KNN: ", knn_acc)
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
perceptron_acc = perceptron.score(X_test, y_test)
print("Accuracy of Perceptron: ", perceptron_acc)
from sklearn.naive_bayes import GaussianNB
gaussiannb = GaussianNB()
gaussiannb.fit(X_train, y_train)
gaussiannb_acc = gaussiannb.score(X_test, y_test)
print("Accuracy of GaussianNB: ", gaussiannb_acc)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_acc = gbc.score(X_test, y_test)
print("Accuracy of GBC: ", gbc_acc)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_acc = xgb.score(X_test, y_test)
print("Accuracy of XGBoost: ", xgb_acc)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
mlp_acc = mlp.score(X_test, y_test)
print("Accuracy of MLP: ", mlp_acc)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)
print("Accuracy of KNN: ", knn_acc)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)
print("Accuracy of Random Forest: ", rf_acc)


