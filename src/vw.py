import pandas as pd
import pyomo.environ as pe
import preprocessing
from optimisation import *
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import model
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import swifter
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from vowpalwabbit import pyvw

# train = pd.read_csv("data/train.csv")
# test = pd.read_csv("data/test.csv")

# from tqdm import tqdm
# def preprocessing_linear(train, test):
#     add_year_dummy = lambda year, num, p: True if year>=(990000 + num * p) and year<=(990000 + num * (p + 1)) else False
#     add_year_dummy_trend = lambda year, num, p: year -(990000 + num * p) - p/2 if year>=(990000 + num * p) and year<=(990000 + num * (p + 1)) else 0

#     columns = [i for i in test.columns if i not in ['galaxy']]

#     for data in [test, train]:
#         for galaxy in tqdm(train['galaxy'].unique()):
#             for i in range(5):
#                 data["galactic year group " + str(i)+' '+galaxy] = data[['galaxy', 'galactic year']].apply(lambda x: add_year_dummy(x['galactic year'], i, 5000) if x['galaxy'] == galaxy else 0, axis=1)
#                 data["galactic year group trend " + str(i)] = data[['galaxy', 'galactic year']].apply(lambda x: add_year_dummy_trend(x['galactic year'], i, 5000) if x['galaxy'] == galaxy else 0, axis=1)
            

#             index_data = data[data.galaxy == galaxy].index
#             index_train = train[train.galaxy == galaxy].index

#             for column in columns:
#                 mean_value = np.mean(train.loc[index_train,column])
#                 data.loc[index_data, column] = data.loc[index_data, column].fillna(mean_value)


#         data = pd.concat([data, pd.get_dummies(train['galaxy'], drop_first=True)], axis=0)
#     return train, test


with open('training_samples', 'rb') as handel:
    training_samples = pickle.load(handel)

with open('validate_samples', 'rb') as handel:
    validate_samples = pickle.load(handel)

with open('train_labels', 'rb') as handel:
    train_labels = pickle.load(handel)

with open('valid_labels', 'rb') as handel:
    valid_labels = pickle.load(handel)



vw = pyvw.vw( 
    loss_function='squared',
    # link='logistic',
    b=25,
    # bootstrap=20,
    q='aa',
    cubic='aaa',
    # ngram=2,
#     skips=1,
#     hash='all',
#     hessian_on=True,
#     random_seed=112,
    l1=0.00001,
    l2=0.00001,
    f='vw.log.model',
    learning_rate=0.005)

for iteration in range(300):
    for i in range(len(training_samples)):
        vw.learn(training_samples[i])
    if iteration % 10 ==0:

        train_predictions = [vw.predict(sample) for sample in training_samples]
        valid_predictions = [vw.predict(sample) for sample in validate_samples]
        print("========== step {0} ==========".format(iteration))
        print("train Loss: ", sqrt(mean_squared_error((train_labels), (train_predictions))))
        print("validate Loss: ", sqrt(mean_squared_error((valid_labels), (valid_predictions))))

vw.finish()