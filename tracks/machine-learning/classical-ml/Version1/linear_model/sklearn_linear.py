import numpy as np
from sklearn.linear_model import LinearRegression
from data_preprocess import x_train, y_train, x_test, y_test

linreg=LinearRegression()
linreg.fit(x_train,y_train)
print('回归系数：', linreg.coef_, linreg.intercept_)

y_pred = linreg.predict(x_test)
rmse = np.sqrt(np.square(y_test - y_pred).mean())
print('RMSE：', rmse)
