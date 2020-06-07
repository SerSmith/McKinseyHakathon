import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import re

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
def preprocessing_all(df_train, df_test, fill_value = -10, coeff = 0.2, num_k = 7):
                      
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

    return df_train, df_test
