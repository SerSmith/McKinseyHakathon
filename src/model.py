import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import optimisation
import random
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from scipy.interpolate import interp1d

# def split_train_val(df_train, df_test, percent_train):
#     # Функция, которая разбивает выборку на трейн и валидацию (которая начаниется с минимального года из теста)
#     date_test_min = min(df_test.galacticyear)
#     k = int(df_train.shape[0] * percent_train)
#     df_train_new = df_train.loc[random.sample(list(df_train.index),k)]
#     df_val_new = df_train[df_train.galacticyear >= date_test_min]
#     return df_train_new.drop('y',axis=1), df_val_new.drop('y',axis=1), df_train_new['y'], df_val_new['y']

def split_train_val(df_train, df_test, percent_val, percent_drop_out):
    # Функция, которая честно разбивает выборку на трейн и валидацию (которая начаниется с минимального года из теста) валидация будет маленькой
    date_test_min = min(df_test.galacticyear)
    df_val = df_train[df_train.galacticyear >= date_test_min]
    k = int(df_val.shape[0] * percent_val)
    df_val_new = df_val.loc[random.sample(list(df_val.index),k)]
    df_train_new = df_train[~df_train.index.isin(df_val_new.index)]
    df_train_new = df_train_new.sample(frac=1-percent_drop_out)
    return df_train_new.drop('y',axis=1), df_val_new.drop('y',axis=1), df_train_new['y'], df_val_new['y']

def train_model_interp(df_train, df_test, percent_val=0.1,percent_drop_out=0.1, y_model=None, previous_residuals=None, round_digits=None):
    X_train, X_val, y_train, y_val = split_train_val(df_train, df_test, percent_val, percent_drop_out)
    for galaxy in tqdm(df_train.galaxy.unique()):
        index = X_train[X_train.galaxy==galaxy].index
        model = interp1d(list(X_train.loc[index,'galacticyear']), list(y_train.loc[index]), kind='linear',fill_value="extrapolate")
        pred = model(X_train[X_train.galaxy==galaxy]['galacticyear'])
        X_val.loc[X_val.galaxy==galaxy,'y_interp'] = model(X_val[X_val.galaxy==galaxy]['galacticyear'])
        X_train.loc[X_train.galaxy==galaxy,'y_interp'] = model(X_train[X_train.galaxy==galaxy]['galacticyear'])
        df_test.loc[df_test.galaxy==galaxy,'y_interp'] = model(df_test[df_test.galaxy==galaxy]['galacticyear'])                
    print(X_train.shape, X_val.shape, min(X_val.galacticyear))
    predict_val = list(X_val['y_interp'])
    predict_test = list(df_test['y_interp'])
    print('RMSE: ', np.sqrt(mean_squared_error(y_val, list(X_val['y_interp']))))

    rank_diviation = None

    if previous_residuals is not None and y_model is not None and round_digits is not None:

        possibal_points = [np.array(previous_residuals) + res for res in predict_val]

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

    return y_val, predict_val, rank_diviation, predict_test

def train_model_lgb(df_train, df_test, trend_y = True, percent_val=0.1, percent_drop_out=0.1, y_model=None, previous_residuals=None, round_digits=None):
    X_train, X_val, y_train, y_val = split_train_val(df_train, df_test, percent_val, percent_drop_out)
    gbm = lgb.LGBMRegressor(objective='rmse', max_depth=12, num_leaves=23, learning_rate=0.01, colsample_bytree=0.800, subsample=0.803, early_stopping_rounds=10, n_estimators=10000)
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

        possibal_points = [np.array(previous_residuals) + res for res in predict]

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

def run_model_and_distrs_interp(train, test, trend_y=False, percent_val=0.1, percent_drop_out=0.1, qunity_starts=1, quantity_points_out=100, edges_percent=0.2, round_digits=6):

    previous_residuals = None
    residuals_all = []
    model_all = []

    rank_diviation_all = []

    y_model = np.linspace(train['y'].min()*(1 - edges_percent), train['y'].max()*(1 + edges_percent), quantity_points_out)

    y_all = np.zeros([test.shape[0], qunity_starts])
    for i in range(qunity_starts):
        y_validate, predict, rank_diviation, y_all[:,i]  = train_model_interp(train, test, percent_val=percent_val, percent_drop_out=percent_drop_out, y_model=y_model, previous_residuals=previous_residuals, round_digits=round_digits)

        residuals = [y - y_true for y, y_true in zip(list(y_validate), predict)]
        previous_residuals = residuals
        residuals_all += residuals
        rank_diviation_all += [rank_diviation]



    if trend_y:
        columns = ['y_trend','galaxy']
        for i, model_out in enumerate(model_all):
            y_all[:, i]+= test['y_trend']
    else:
        columns = ['galaxy']

    mean_y = y_all.mean(axis=1)

    possibal_points = [np.array(residuals_all) + res for res in mean_y]

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

def run_model_and_distrs(train, test, trend_y=False, percent_val=0.1, percent_drop_out=0.1, qunity_starts=1, quantity_points_out=100, edges_percent=0.2, round_digits=6):

    previous_residuals = None
    residuals_all = []
    model_all = []

    rank_diviation_all = []

    y_model = np.linspace(train['y'].min()*(1 - edges_percent), train['y'].max()*(1 + edges_percent), quantity_points_out)

    y_all = np.zeros([test.shape[0], qunity_starts])
    for i in range(qunity_starts):
        X_train, X_validate, y_train, y_validate, model_out, predict, rank_diviation  = train_model_lgb(train, test, trend_y, percent_val=percent_val, percent_drop_out=percent_drop_out, y_model=y_model, previous_residuals=previous_residuals, round_digits=round_digits)

        residuals = [y - y_true for y, y_true in zip(list(y_validate), predict)]
        previous_residuals = residuals
        residuals_all += residuals
        model_all += [model_out]
        rank_diviation_all += [rank_diviation]



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

    possibal_points = [np.array(residuals_all) + res for res in mean_y]

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
