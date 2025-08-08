from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from data_util import get_reg_data
import numpy as np
import xgboost as xgb

x_train, x_test, y_train, y_test = get_reg_data()

gbdt = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbdt.fit(x_train, y_train)
print("GBDT RMSE:", np.sqrt(mean_squared_error(y_test, gbdt.predict(x_test))))

xgbr = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=42
)
xgbr.fit(x_train, y_train)
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, xgbr.predict(x_test))))