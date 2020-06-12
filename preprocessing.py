import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import re
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from collections import defaultdict

class always_zero:
    '''Класс - заглушка, возвращающая всегда 0
    '''
    def predict(self, x):
        return np.zeros_like(x)

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
        df_galactic = df_all[df_all.galaxy == galactic].sort_values(['galacticyear']).copy()
        df_galactic = pd.concat([df_galactic, df_galactic['y'].shift(1)], axis=1)
        df_galactic.columns = list(df_all.columns) + ['y_shift']
        df_galactic['y_shift'].iloc[0] = np.mean(df_galactic.y_shift)
        for i,j in enumerate(df_galactic.y_shift):
            if np.isnan(j):
                df_galactic['y_shift'].iloc[i] = df_galactic['y_shift'].iloc[i-1]
        df_train_new = pd.concat([df_train_new, df_galactic[df_galactic.train == 1].drop(['train'], axis=1)], axis=0)
        df_test_new = pd.concat([df_test_new, df_galactic[df_galactic.train == 0].drop(['train'], axis=1)], axis=0)
    df_train_new.loc[df_train_new.galaxy=='NGC 5253','y_shift'] = df_train_new[df_train_new.galaxy=='NGC 5253']['y'].iloc[0]
    return df_train_new, df_test_new

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
    for column in columns:
        if column in df_train.columns:
            df_train[column] = df_train[column].fillna(value)
    for column in columns:
        if column in df_test.columns:
            df_test[column] = df_test[column].fillna(value)  
    return df_train, df_test

# Функция, которая объединяет в себе весь препроцессинг
def preprocessing_all(df_train, df_test, trend_features = False, trend_y = False, fill_value = -10, coeff = 0.2, num_k = 7):
    
    if trend_features:
        df_train, df_test, models_dict = delete_trend(df_train, df_test)

    # df_train.to_excel('tmp1.xlsx')
    # df_test.to_excel('tmp2.xlsx')

    columns = [i for i in df_test.columns if i not in ['galaxy']]
    for galaxy in df_train.galaxy.unique():
        index_train = df_train[df_train.galaxy == galaxy].index
        index_test = df_test[df_test.galaxy == galaxy].index
        for column in columns:
            mean_value = np.mean(df_train.loc[index_train,column])
            df_train.loc[index_train, column] = df_train.loc[index_train, column].fillna(mean_value)            
            df_test.loc[index_test, column] =  df_test.loc[index_test, column].fillna(mean_value)
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
    df_train, df_test = fillna(df_train, df_test, columns, fill_value)
    df_train['galaxy'] = df_train_galaxy
    df_test['galaxy'] = df_test_galaxy

    df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df_train, df_test = add_y_shift(df_train, df_test)
    df_test.sort_index(inplace=True)
    df_train.sort_index(inplace=True)
    
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


    return df_train, df_test

if __name__ == '__main__':

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train_upd, test_upd = preprocessing_all(train, test)