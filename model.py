import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats

def train_model(df_train, percent_val=0.2):
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop('y', axis=1), df_train['y'], test_size=percent_val)
    gbm = lgb.LGBMRegressor(objective='rmse',max_depth=5, num_leaves=26, learning_rate=0.060, colsample_bytree=0.800,
                            subsample=0.968, n_estimators=451)
    gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='RMSE')
    predict = gbm.predict(X_test)
    return X_train, X_test, y_train, y_test, gbm, predict 


def run_model_and_distrs(train, test, percent_val=0.2, qunity_starts=1, quantity_points_out=100, edges_percent=0.2, round_digits=6):

    residuals_all = []
    model_all = []
    for i in range(qunity_starts):
        X_train, X_validate, y_train, y_validate, model_out, predict = train_model(train, percent_val=percent_val)

        residuals = [y - y_true for y, y_true in zip(list(y_validate), predict)]
        residuals_all += residuals
        model_all += [model_out]

    y_all = np.zeros([test.shape[0], qunity_starts])

    for i, model_out in enumerate(model_all):
        y = model_out.predict(test)
        y_all[:, i] = y

    mean_y = y_all.mean(axis=1)

    possibal_points = [np.array(residuals_all) + res for res in y]

    kernels_all = []

    for points in possibal_points:
        kernel = stats.gaussian_kde(points)
        kernels_all += [kernel]

    y_model = np.linspace(train['y'].min()*(1 - edges_percent), train['y'].max()*(1 + edges_percent), quantity_points_out)

    y_prob_model = np.zeros([test.shape[0], y_model.shape[0]])

    for i, kernel in enumerate(kernels_all):
        y_prob_model[i, :] = kernel(y_model)

    rounded_y_prob_model = np.round(y_prob_model, round_digits)
    y_prob_model = rounded_y_prob_model/rounded_y_prob_model.sum(axis=1)[:, None]

    test['y'] = mean_y

    return test, y_prob_model, y_model
