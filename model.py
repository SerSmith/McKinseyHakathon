import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import optimisation
import random

def split_train_val(df_train, df_test, percent_train):
    # Функция, которая разбивает выборку на трейн и валидацию (которая начаниется с минимального года из теста)
    date_test_min = min(df_test.galacticyear)
    k = int(df_train.shape[0] * percent_train)
    df_train_new = df_train.loc[random.sample(list(df_train.index),k)]
    df_val_new = df_train[df_train.galacticyear >= date_test_min]
    return df_train_new.drop('y',axis=1), df_val_new.drop('y',axis=1), df_train_new['y'], df_val_new['y']
 

def train_model(df_train, df_test, trend_y = True, percent_train=0.8, y_model=None, previous_residuals=None, round_digits=None):
    X_train, X_val, y_train, y_val = split_train_val(df_train, df_test, percent_train)
    gbm = lgb.LGBMRegressor(objective='rmse', max_depth=5, num_leaves=26, learning_rate=0.060, colsample_bytree=0.800,
                            subsample=0.968, n_estimators=451)
    print(X_train.shape, X_val.shape, min(X_val.galacticyear))
    if trend_y:
        columns=['y_trend','galaxy']
    else:
        columns=['galaxy']
    gbm.fit(X_train.drop(columns=columns), y_train,
        eval_set=[(X_val.drop(columns=columns), y_val)],
        eval_metric='RMSE')
    predict = gbm.predict(X_val.drop(columns=columns))

    
    if trend_y:
        predict += X_val['y_trend']
        y_train += X_train['y_trend']
        y_val += X_val['y_trend']


    rank_diviation = None

    if previous_residuals is not None and y_model is not None and round_digits is not None:

        possibal_points = [np.array(previous_residuals) + res for res in y_val]

        kernels_all = []

        for points in possibal_points:
            kernel = stats.gaussian_kde(points)
            kernels_all += [kernel]


        y_prob_model = np.zeros([y_val.shape[0], y_model.shape[0]])

        for i, kernel in enumerate(kernels_all):
            y_prob_model[i, :] = kernel(y_model)

        rounded_y_prob_model = np.round(y_prob_model, round_digits)
        y_prob_model = rounded_y_prob_model/rounded_y_prob_model.sum(axis=1)[:, None]


        probs = optimisation.add_probs(y_prob_model, y_model)
        
        probs_rank = probs.sum(axis=0).argsort()/probs.shape[0]

        y_val_rank = y_val.argsort()/probs.shape[0]
        
        rank_diviation = np.sum((probs_rank - y_val_rank) ** 2)/y_val_rank.shape[0]



    return X_train, X_val, y_train, y_val, gbm, predict, rank_diviation


def run_model_and_distrs(train, test, trend_y=True, percent_train=0.8, qunity_starts=1, quantity_points_out=100, edges_percent=0.2, round_digits=6):

    previous_residuals = None
    residuals_all = []
    model_all = []

    rank_diviation_all = []

    y_model = np.linspace(train['y'].min()*(1 - edges_percent), train['y'].max()*(1 + edges_percent), quantity_points_out)


    for i in range(qunity_starts):
        X_train, X_validate, y_train, y_validate, model_out, predict, rank_diviation  = train_model(train, test, trend_y, percent_train=percent_train, y_model=y_model, previous_residuals=previous_residuals, round_digits=round_digits)

        residuals = [y - y_true for y, y_true in zip(list(y_validate), predict)]
        previous_residuals = residuals
        residuals_all += residuals
        model_all += [model_out]
        rank_diviation_all += [rank_diviation]

    y_all = np.zeros([test.shape[0], qunity_starts])

    if trend_y:
        columns = ['y_trend','galaxy']
        for i, model_out in enumerate(model_all):
            y_all[:, i]+= test['y_trend']
    else:
        columns = ['galaxy']
    for i, model_out in enumerate(model_all):
        y = model_out.predict(test.drop(columns=columns))
        y_all[:, i] = y

    mean_y = y_all.mean(axis=1)

    possibal_points = [np.array(residuals_all) + res for res in y]

    kernels_all = []

    for points in possibal_points:
        kernel = stats.gaussian_kde(points)
        kernels_all += [kernel]


    y_prob_model = np.zeros([test.shape[0], y_model.shape[0]])

    for i, kernel in enumerate(kernels_all):
        y_prob_model[i, :] = kernel(y_model)

    rounded_y_prob_model = np.round(y_prob_model, round_digits)
    y_prob_model = rounded_y_prob_model/rounded_y_prob_model.sum(axis=1)[:, None]

    test['y'] = mean_y

    rank_diviation_all = rank_diviation_all[1:]

    return test, y_prob_model, y_model, rank_diviation_all
