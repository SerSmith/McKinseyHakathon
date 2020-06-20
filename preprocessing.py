import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import re
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
from tslearn.clustering import TimeSeriesKMeans, to_time_series_dataset

class always_zero:
    '''Класс - заглушка, возвращающая всегда 0
    '''
    def predict(self, x):
        return np.zeros_like(x)




def preprocessing_add_epoch_statistics(train, test):
    
    
    # add_year_dummy = lambda year, num, p: True if year>=(990000 + num * p) and year<=(990000 + num * (p + 1)) else False

    train['epoch'] = train['galactic year'].apply(lambda x: int((x-990000)/5000))
    train['ten_years'] = train['galactic year'].apply(lambda x: int((x-990000 - int((x-990000)/5000)*5000)/1000))
    train = pd.concat([train, pd.get_dummies(train['epoch'], drop_first=True)], axis=1)

    test['epoch'] = test['galactic year'].apply(lambda x: int((x-990000)/5000))
    test['ten_years'] = test['galactic year'].apply(lambda x: int((x-990000 - int((x-990000)/5000)*5000)/1000))
    test = pd.concat([test, pd.get_dummies(test['epoch'], drop_first=True)], axis=1)

    # Оставим толь колонки которые есть в тесте
    train = train.loc[:, list(test.columns) + ['y']]
    return train, test

def delete_trend(df_train, df_test):
    models_dict=defaultdict(dict)
    for column in tqdm(df_train.columns):
        if column not in ['galactic year', 'galaxy']:
            for galaxy in df_train['galaxy'].unique():
                index_train = df_train[(df_train.galaxy == galaxy) & (df_train[column].notnull())].index
                y = df_train.loc[index_train, column].to_numpy().reshape(-1, 1)
                X = df_train.loc[index_train, 'galactic year'].to_numpy().reshape(-1, 1)
                if len(y):
                    model = LinearRegression().fit(X, y)
                else:
                    model = always_zero()

                models_dict[column][galaxy] = model

                if len(y):
                    trend = pd.Series(models_dict[column][galaxy].predict(X).reshape(-1))
                    trend.index = index_train
                    df_train.loc[index_train, column] = df_train.loc[index_train, column] - trend

    for column in tqdm(df_test.columns):
        if column not in ['galactic year', 'galaxy']:
            for galaxy in df_test['galaxy'].unique():
                index_test = df_test[(df_test.galaxy == galaxy) & (df_train[column].notnull())].index
                y = df_test.loc[index_test, column].to_numpy().reshape(-1, 1)
                X = df_test.loc[index_test, 'galactic year'].to_numpy().reshape(-1, 1)
                if len(X):
                    trend = pd.Series(models_dict[column][galaxy].predict(X).reshape(-1))
                    trend.index = index_test
                    df_test.loc[index_test, column] =  trend
        
    return df_train, df_test, models_dict




def add_y_shift(df_train, df_test):
    df_train_new = pd.DataFrame()
    df_test_new = pd.DataFrame()
    df_train['train'] = 1
    df_test['train'] = 0
    df_all = pd.concat([df_train,df_test])
    for galactic in df_train.galaxy.unique():  
        df_galactic = df_all[df_all.galaxy == galactic].sort_values(['galactic year']).copy()
        df_galactic = pd.concat([df_galactic, df_galactic['y'].shift(1)], axis=1)
        df_galactic.columns = list(df_all.columns) + ['y_shift']
        df_galactic['y_shift'].iloc[0] = np.mean(df_galactic.y_shift)
        for i,j in enumerate(df_galactic.y_shift):
            if np.isnan(j):
                df_galactic['y_shift'].iloc[i] = df_galactic['y_shift'].iloc[i-1]
        df_train_new = pd.concat([df_train_new, df_galactic[df_galactic.train == 1].drop(['train'], axis=1)], axis=0)
        df_test_new = pd.concat([df_test_new, df_galactic[df_galactic.train == 0].drop(['train'], axis=1)], axis=0)
    df_train_new.loc[df_train_new.galaxy=='NGC 5253','y_shift'] = df_train_new[df_train_new.galaxy=='NGC 5253']['y'].iloc[0] # Выделила отдельно галактику, так как по ней всего 1 строчка
    return df_train_new, df_test_new.drop('y', axis=1)

def my_add_feature(df_train, df_test, columns, fill_value, coef, num_k):
    edge_right= defaultdict()
    edge_left = defaultdict()

    for column in columns:

        df_col = df_train[df_train[column].isnull()==False][[column,'y']]
        st = stats.binned_statistic(df_col[column], df_col['y'], statistic='mean', bins=20)

        # Ищем фичи и границу для них, распределение которых выглядит как нижний правый угол
        min_value = st.statistic[0] * (1 - coef)
        max_value = st.statistic[0] * (1 + coef)
        k=0
        mask = ~np.isnan(st.statistic)
        for value, edge in zip(st.statistic[mask], st.bin_edges[:-1][mask]):
            if (value>=min_value) & (value<=max_value):
                k+=1
            else:
                if k>num_k:
                    #print(k, round(edge,2),'right', column)
                    edge_right[column] = edge
                break
        # Ищем фичи и границу для них, распределение которых выглядит как нижний левый угол        
        min_value = st.statistic[-1] * (1 - coef)
        max_value = st.statistic[-1] * (1 + coef)
        k=0
        mask = ~np.isnan(st.statistic)
        for value, edge in zip(st.statistic[mask][::-1], st.bin_edges[1:][mask][::-1]):
            if (value >= min_value) & (value <= max_value):
                k+=1
            else:
                if k>num_k:
                    #print(k, np.round(edge,2), 'left', column)
                    edge_left[column] = edge
                break        
    # Теперь создаём новые фичи на основе отобранных             
    for column in edge_left.keys():
        column1 = 'const__' + column
        column2 = 'varios__' + column
        column3 = 'nan__' + column
        df_train[column1] = df_train[column].apply(lambda x: 1 if x >= edge_left[column] else 0)
        df_train[column2] = df_train[column].apply(lambda x: x if x < edge_left[column] else fill_value)
        df_train[column3] = df_train[column].apply(lambda x: 1 if np.isnan(x) else 0)
        #df_train = df_train.drop(column, axis=1)
    for column in edge_right.keys():
        column1 = 'const__' + column
        column2 = 'varios__' + column
        column3 = 'nan__' + column
        df_train[column1] = df_train[column].apply(lambda x: 1 if x <= edge_right[column] else 0 )
        df_train[column2] = df_train[column].apply(lambda x: x if x > edge_right[column] else fill_value)
        df_train[column3] = df_train[column].apply(lambda x: 1 if np.isnan(x) else 0)
        #df_train = df_train.drop(column, axis=1)

    # Для тестовой выборки проделаем то же самое,  с уже имеющимися границами    
    for column in edge_left.keys():
        column1 = 'const__' + column
        column2 = 'varios__' + column
        column3 = 'nan__' + column
        df_test[column1] = df_test[column].apply(lambda x: 1 if x >= edge_left[column] else 0)
        df_test[column2] = df_test[column].apply(lambda x: x if x < edge_left[column] else fill_value)
        df_test[column3] = df_test[column].apply(lambda x: 1 if np.isnan(x) else 0)
        #df_train = df_train.drop(column, axis=1)
    for column in edge_right.keys():
        column1 = 'const__' + column
        column2 = 'varios__' + column
        column3 = 'nan__' + column
        df_test[column1] = df_test[column].apply(lambda x: 1 if x <= edge_right[column] else 0 )
        df_test[column2] = df_test[column].apply(lambda x: x if x > edge_right[column] else fill_value)
        df_test[column3] = df_test[column].apply(lambda x: 1 if np.isnan(x) else 0)
        #df_train = df_train.drop(column, axis=1)    
    return  df_train, df_test, edge_right, edge_left





def fillna(df_train, df_test, columns, value):



    appended = df_train.append(df_test, ignore_index=True)

    columns = [i for i in appended.columns if i not in ['galaxy', 'galactic year', 'y']]

    for galaxy in tqdm(appended.galaxy.unique()):
        for column in columns:
            appended_index = appended[(appended.galaxy == galaxy) & (appended[column].notnull())].index
            index_train = df_train[(appended.galaxy == galaxy) & df_train[column].isnull()].index
            index_test = df_test[(appended.galaxy == galaxy) & df_test[column].isnull()].index
            if (index_train.shape[0] + index_test.shape[0]) > 0 and (appended_index.shape[0]) > 2:
                year = appended.loc[appended_index, 'galactic year'].to_numpy()
                column_to_predict = appended.loc[appended_index, column].to_numpy()
                model = interp1d(year, column_to_predict, kind='linear',fill_value="extrapolate")
                df_train.loc[index_train, column] = model(df_train.loc[index_train, 'galactic year'].to_numpy().reshape(-1, 1))
                df_test.loc[index_test, column] = model(df_test.loc[index_test, 'galactic year'].to_numpy().reshape(-1, 1)) 


    for column in columns:
        if column in df_train.columns:
            df_train[column] = df_train[column].fillna(value)
    for column in columns:
        if column in df_test.columns:
            df_test[column] = df_test[column].fillna(value)  
    return df_train, df_test

def add_features_from_regression(df_train, df_test):
    models_dict=defaultdict(dict)
    column = 'y'
    df_train[f'regres_{column}'] =  0
    df_train[f'coef_regr_{column}'] = 0
    df_train[f'inter_regr__{column}'] = 0
#     for column in tqdm(df_train.columns):
#         df_train[f'regres_{column}'] =  0
#         df_train[f'coef_regr_{column}'] = 0
#         df_train[f'inter_regr__{column}'] = 0
#         if column not in ['galactic year', 'galaxy']:
    for galaxy in df_train['galaxy'].unique():
        index_train = df_train[(df_train.galaxy == galaxy) & (df_train[column].notnull())].index
        y = df_train.loc[index_train, column].to_numpy().reshape(-1, 1)
        X = df_train.loc[index_train, 'galactic year'].to_numpy().reshape(-1, 1)
        if len(y):
            model = LinearRegression().fit(X, y)
        else:
            model = always_zero()

        models_dict[galaxy] = model

        if len(y):
            index_train = df_train[df_train.galaxy == galaxy].index
            X = df_train.loc[index_train, 'galactic year'].to_numpy().reshape(-1, 1)
            trend = pd.Series(models_dict[galaxy].predict(X).reshape(-1))
            trend.index = index_train
            df_train.loc[index_train, f'regres_{column}'] =  trend
            df_train.loc[index_train, f'coef_regr_{column}'] = models_dict[galaxy].coef_[0][0]*1000000 # Добавили умножение на 100000, чтобы число было не слишком маленьким
            df_train.loc[index_train, f'inter_regr_{column}'] = models_dict[galaxy].intercept_[0]



#     for column in tqdm(df_test.columns):
#         df_test[f'regres_{column}'] =  0
#         df_test[f'coef_regr_{column}'] = 0
#         df_test[f'inter_regr__{column}'] = 0
#         if column not in ['galactic year', 'galaxy']:
    for galaxy in df_train['galaxy'].unique():
        index_test = df_test[df_test.galaxy == galaxy].index
        X = df_test.loc[index_test, 'galactic year'].to_numpy().reshape(-1, 1)
        if len(X):
            trend = pd.Series(models_dict[galaxy].predict(X).reshape(-1))
            trend.index = index_test
            df_test.loc[index_test, f'regres_{column}'] =  trend
            df_test.loc[index_test, f'coef_regr_{column}'] = models_dict[galaxy].coef_[0][0]*1000000
            df_test.loc[index_test, f'inter_regr__{column}'] = models_dict[galaxy].intercept_[0]
    return df_train, df_test 


def add_label_from_clusterization(df_train, df_test):
    y_galaxy = []
    for galactic in df_train.galaxy.unique():  
        y_galaxy.append(df_train[df_train.galaxy == galactic].sort_values(['galactic year']).y.values)
    X_galaxy = to_time_series_dataset(y_galaxy)
    km_dba = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=10,
                              max_iter_barycenter=5,
                              random_state=0).fit(X_galaxy)
    labels = list(km_dba.labels_)
    df_train['label_cluster'] = 0
    df_test['label_cluster'] = 0
    for i, galactic in enumerate(df_train.galaxy.unique()):  
        df_train[df_train.galaxy == galactic]['label_cluster'] = labels[i]
        df_test[df_test.galaxy == galactic]['label_cluster'] = labels[i]
    return df_train, df_test

# Функция, которая объединяет в себе весь препроцессинг
def preprocessing_all(df_train, df_test, trend_features = False, trend_y = False, fill_value = -10, coeff = 0.2, num_k = 7):
    
    true_existence_expectancy_index = df_test['existence expectancy index']

    if trend_features:
        df_train, df_test, models_dict = delete_trend(df_train, df_test)

    df_train, df_test = add_features_from_regression(df_train, df_test)
    columns = [i for i in df_test.columns if i not in ['galaxy']]
    df_train, df_test = add_label_from_clusterization(df_train, df_test)   

    galaxy_train = df_train['galaxy'].unique()
    galaxy_test = df_test['galaxy'].unique()
    df_train_galaxy = df_train['galaxy']
    df_test_galaxy = df_test['galaxy']
    # Получим колонки - galaxy_del, которые нужно выкинуть из train, так как их не будет в тесте
    galaxy_del = list(set(galaxy_train)- set(galaxy_test))
    galaxy_del = ['galaxy_' + name for name in galaxy_del] 
    # one hot encoding переменной galaxy
    df_train = pd.get_dummies(df_train, columns=['galaxy'])
    df_test = pd.get_dummies(df_test, columns=['galaxy'])
    df_train = df_train.drop(galaxy_del, axis=1)
    
    df_train, df_test, edge_right, edge_left = my_add_feature(df_train, df_test, columns, fill_value, coeff, num_k)

    df_train['galaxy'] = df_train_galaxy
    df_test['galaxy'] = df_test_galaxy



    df_train, df_test = preprocessing_add_epoch_statistics(df_train, df_test)

    if trend_y:
        column = 'y'

        for galaxy in df_train['galaxy'].unique():
            index_train = df_train[(df_train.galaxy == galaxy)].index
            X = df_train.loc[index_train, 'galacticyear'].to_numpy().reshape(-1, 1)
            trend = pd.Series(models_dict[column][galaxy].predict(X).reshape(-1))
            trend.index = index_train
            df_train.loc[index_train, 'y_trend'] = trend


        for galaxy in df_test['galaxy'].unique():
            index_test = df_test[df_test.galaxy == galaxy].index
            X = df_test.loc[index_test, 'galacticyear'].to_numpy().reshape(-1, 1)
            trend = pd.Series(models_dict[column][galaxy].predict(X).reshape(-1))
            trend.index = index_test
            df_test.loc[index_test, 'y_trend'] = trend



    df_train, df_test = add_y_shift(df_train, df_test)
    
    df_test.sort_index(inplace=True)
    df_train.sort_index(inplace=True)
    
    df_train, df_test = fillna(df_train, df_test, columns, fill_value)
    
    df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', str(x)))
    df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', str(x)))

    return df_train, df_test, true_existence_expectancy_index

if __name__ == '__main__':

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train_upd, test_upd, true_existence_expectancy_index = preprocessing_all(train, test)
    # preprocessing_add_epoch_statistics(train, test)