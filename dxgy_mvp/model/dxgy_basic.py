"""
    0. Configuration
"""
file_dir = '/Users/skhalilbekov/Desktop/wokrflow/dxgy_mvp'
sql_query_path = '/sql/get_dxgy_data.sql'
spends_query_path = 'sql/get_spend_shares.sql'
locality = 22534

target_cols = ['EXPECTED_RIDES']
drop_cols_for_train = ['ID', 'SPLIT_NO', 'PERIOD_DATE_START']
unique_obs_groupper = ['ID', 'SPLIT_NO', 'PERIOD_DATE_START', 'SEGMENT_TYPE']


params_grid = {
    'max_depth': [2, 3, 4],
    'n_estimators': [50, 100, 500],
    'max_features': [3, 4, 5]
    }

"""
    1. Modules and functions
"""
%load_ext dotenv
%dotenv
%reload_ext dotenv

%load_ext autoreload
%autoreload 2

import os
import pyexasol
import pandas as pd
from numpy import mean
from numpy import std
from numpy import absolute
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder

from utils.utils import *

pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 10)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def get_dxgy_data(file_dir: str, sql_query_path: str, locality: int):
    
    query = sql_from_file(file_dir + sql_query_path).format(locality)
    db_host = pyexasol.connect(dsn = 'ex1..3.city-srv.ru:8563',
                              user = os.getenv('MAIN_USER'),
                              password = os.getenv('PASSWORD'), 
                              fetch_dict=True)

    df = db_host.export_to_pandas(query)
    
    return df

def is_weekend(day):
    if day <= 3:
        is_weekend = 0
    elif 3 < day < 7:
        is_weekend = 1
    else:
        is_weekend = -1
    return is_weekend

def calc_time_features(df: pd.DataFrame):
    
    df['month'] = df['PERIOD_DATE_START'].dt.month
    df['quarter'] = df['PERIOD_DATE_START'].dt.quarter
    df['dayofweek'] = df['PERIOD_DATE_START'].dt.dayofweek
    df['dayname'] = df['PERIOD_DATE_START'].dt.day_name()
    df['is_weekend'] = df['dayofweek'].apply(is_weekend)
    
    return df
    
def get_feature_importances(forest, X: pd.DataFrame):
    
    feature_names = [f"{i}" for i in X.columns]
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

# кодируем кат фичи
def code_dummy(data, feature):
    for i in data[feature].unique():
        data[feature + '=' + str(i)] = (data[feature] == i).astype(float)

"""
    2. Main
"""
# raw data
df = get_dxgy_data(file_dir, sql_query_path, locality)

# group by bonus id and split
main_df = df.groupby(unique_obs_groupper).agg({'DRIVER_ID': 'count',
                                               'BUDGET': 'sum',
                                               'PERIOD': 'mean',
                                               'FACT_RIDES': 'sum',
                                               'TARGET_Y': 'mean',
                                               'TARGET_X': 'mean'})\
                      .rename(columns = {'DRIVER_ID': 'total_drivers_count',
                                         'FACT_RIDES': 'EXPECTED_RIDES'})\
                      .reset_index()
filtered_main_df = main_df.loc[main_df['PERIOD_DATE_START'] < '2021-07-01 00:00:00']
filtered_main_df['TARGET_Y'] = filtered_main_df['BUDGET'] / filtered_main_df['total_drivers_count']
#filtered_main_df = filtered_main_df.loc[filtered_main_df['TARGET_Y'] > 1000]

y, X = filtered_main_df[target_cols], filtered_main_df.drop(target_cols + drop_cols_for_train, axis = 1)

# использование случайного леса сразу по трем таргетам вместе
forest = RandomForestRegressor(random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,
                                                    train_size = .2)
rf_grid = GridSearchCV(estimator = forest, param_grid = params_grid, cv = 3)
rf_grid.fit(X_train, y_train)

# best params
rf_params = rf_grid.best_params_
print(f'Best cross val scoore: {rf_grid.best_score_}')

# fit best params
rf = RandomForestRegressor(n_estimators = rf_params['n_estimators'], 
                           max_depth = rf_params['max_depth'] - 1,
                           max_features = rf_params['max_features'],
                           random_state = 42)
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))

valid = pd.DataFrame({'SEGMENT_TYPE': [0],
                      'total_drivers_count': [534],
                      'BUDGET': [320400],
                      'PERIOD': [1],
                      'TARGET_Y': [600],
                      'TARGET_X': [6]})
print(f'Expected num of rides: {rf.predict(valid[X_test.columns])}\nExpected spend: {rf.predict(valid[X_test.columns]) / 6 * 600}')

get_feature_importances(rf, X_train)




"""
    ADHOCs
"""

"""
filtered_main_df['di_per_trip'] = filtered_main_df['TARGET_Y'] / filtered_main_df['TARGET_X']
filtered_main_df = filtered_main_df.reset_index(drop = True)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 12))
ax.scatter(filtered_main_df['di_per_trip'], filtered_main_df['EXPECTED_RIDES'],
            s = filtered_main_df['total_drivers_count']);
n = filtered_main_df['TARGET_X'].astype(str) + '_' + filtered_main_df['TARGET_Y'].astype(str)
for i, txt in enumerate(n):
    ax.annotate(txt, (filtered_main_df.loc[i, 'di_per_trip'],
                      filtered_main_df.loc[i, 'EXPECTED_RIDES']), fontsize = 12)

plt.xlabel('DxGy per Trip')
plt.ylabel('Total Rides')
"""

