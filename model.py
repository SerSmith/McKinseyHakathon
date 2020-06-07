import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_model(df_train, percent_val = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop('y', axis=1), df_train['y'], test_size=percent_val)
    gbm = lgb.LGBMRegressor(objective='regression',num_leaves=12,
                              learning_rate=0.01, n_estimators = 1000)
    gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='RMSE',
        early_stopping_rounds=10)
    predict = gbm.predict(X_test)
    return X_train, X_test, y_train, y_test, gbm, predict 