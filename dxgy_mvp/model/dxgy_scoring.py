#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 07:55:52 2021

@author: skhalilbekov
"""

"""
    0. Configuration
"""
file_dir = '/Users/skhalilbekov/Desktop/wokrflow/dxgy_mvp'
sql_setups_query_path = '/sql/get_dxgy_data.sql'
sql_context_query_path = '/sql/get_city_context.sql'
sql_drivers_context_path = '/sql/get_drivers_context.sql'
valid_driver_ids = '/data/drivers_to_reactivate_2_10_weeks.csv'
locality = 22534
start_date = '2020-04-01'
end_date = '2021-12-01'

unique_obs_groupper = ['id', 'split_no', 'period_date_start',
                       'segment_type', 'driver_id']

params_grid = {
    'max_depth': [3, 4],
    'n_estimators': [100, 500],
    'max_features': [4, 5]
    }

final_features_list = ['target_x', 'budget', 'period', 'trips',
                       'days_since_last_trip', 'days_since_first_trip']

"""
    1. Modules and functions
"""
%load_ext dotenv
%dotenv
%reload_ext dotenv

import os
import pyexasol
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, plot_roc_curve, f1_score
from imblearn.under_sampling import RandomUnderSampler

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

def get_drivers_context(file_dir: str, sql_drivers_context_path: str,
                        locality: int, end_date: str):
    
    query = sql_from_file(file_dir + sql_drivers_context_path)\
                    .format(locality, end_date)

    db_host = pyexasol.connect(dsn = 'ex1..3.city-srv.ru:8563',
                              user = os.getenv('MAIN_USER'),
                              password = os.getenv('PASSWORD'), 
                              fetch_dict=True)

    df = db_host.export_to_pandas(query)

    return df

def prepare_main_df_for_trips_prediction(historical_dxgy_setups: pd.DataFrame,
                                         drivers_context: pd.DataFrame):
        
    historical_dxgy_setups['PERIOD_DATE_START'] = historical_dxgy_setups['PERIOD_DATE_START']\
                                                  .apply(lambda x: pd.to_datetime(str(x)[:10]))

    drivers_context['DT_FIRST'] = pd.to_datetime(drivers_context['DT_FIRST'])
    drivers_context['DT_LAST'] = pd.to_datetime(drivers_context['DT_LAST'])

    main_trips_df = pd.merge(historical_dxgy_setups, drivers_context, how = 'left',
                             on = 'DRIVER_ID')
    main_trips_df.columns = [x.lower() for x in main_trips_df.columns]
    
    # define target (1 if reached the goal, 0 otherwise)
    main_trips_df.loc[main_trips_df['fact_rides'] > 0, 'activated'] = 1
    main_trips_df.loc[main_trips_df['fact_rides'] == 0, 'activated'] = 0
        
    main_trips_df['days_since_last_trip'] = (main_trips_df['period_date_start'] - main_trips_df['dt_last'])\
                                            .dt.days.clip(lower = 0)
    main_trips_df['days_since_first_trip'] = (main_trips_df['period_date_start'] - main_trips_df['dt_first'])\
                                            .dt.days.clip(lower = 0)

    main_trips_df.drop(['target_y', 'fact_rides', 'dt_last', 'dt_first'], axis = 1, inplace = True)
    main_trips_df.fillna(0, inplace = True)

    return main_trips_df
    

def prepare_main_df(historical_dxgy_setups: pd.DataFrame,
                    drivers_context: pd.DataFrame):
    
    historical_dxgy_setups['PERIOD_DATE_START'] = historical_dxgy_setups['PERIOD_DATE_START']\
                                                  .apply(lambda x: pd.to_datetime(str(x)[:10]))
    drivers_context['DT_FIRST'] = pd.to_datetime(drivers_context['DT_FIRST'])
    drivers_context['DT_LAST'] = pd.to_datetime(drivers_context['DT_LAST'])

    main_df = pd.merge(historical_dxgy_setups, drivers_context, how = 'left', on = 'DRIVER_ID')
    main_df.columns = [x.lower() for x in main_df.columns]
    
    # define target (1 if reached the goal, 0 otherwise)
    main_df.loc[main_df['target_y'].isnull(), 'reached_goal'] = 0
    main_df.loc[~main_df['target_y'].isnull(), 'reached_goal'] = 1
        
    main_df['days_since_last_trip'] = (main_df['period_date_start'] - main_df['dt_last']).dt.days.clip(lower = 0)
    main_df['days_since_first_trip'] = (main_df['period_date_start'] - main_df['dt_first']).dt.days.clip(lower = 0)

    main_df.drop(['target_y','dt_last', 'dt_first'], axis = 1, inplace = True)

    return main_df

def calc_expected_rides_and_spends(valid_df: pd.DataFrame, commission_rate = .19):
    
    output = {'trips': np.sum(valid_df['calibrated_preds'] * valid_df['target_x']),
              'spends': np.sum(valid_df['calibrated_preds'] * valid_df['budget']) * (1 - commission_rate)}
    
    return output

def prepare_valid_set(driver_ids: pd.DataFrame, sql_setups_query_path: str,
                      sql_drivers_context_path: str, start_date = '2021-11-24',
                      setup_id = 45519):

    historical_dxgy_setups = get_dxgy_data(file_dir, sql_setups_query_path, locality)
    historical_dxgy_setups = historical_dxgy_setups.loc[historical_dxgy_setups['ID'] == setup_id]
    historical_dxgy_setups['PERIOD_DATE_START'] = pd.to_datetime(historical_dxgy_setups['PERIOD_DATE_START'])
    historical_dxgy_setups = historical_dxgy_setups.loc[historical_dxgy_setups['PERIOD_DATE_START'] == pd.to_datetime(start_date)]
    drivers_context = get_drivers_context(file_dir, sql_drivers_context_path,
                                          locality, start_date)
    valid_df = prepare_valid_df(historical_dxgy_setups, drivers_context)
    
    return valid_df
    
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

def get_optimal_threshold(target: np.array, preds: np.array):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, preds)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0]

def prepare_valid_df(historical_dxgy_setups: pd.DataFrame,
                    drivers_context: pd.DataFrame):
    
    historical_dxgy_setups['PERIOD_DATE_START'] = historical_dxgy_setups['PERIOD_DATE_START']\
                                                  .apply(lambda x: pd.to_datetime(str(x)[:10]))

    drivers_context['DT_FIRST'] = pd.to_datetime(drivers_context['DT_FIRST'])
    drivers_context['DT_LAST'] = pd.to_datetime(drivers_context['DT_LAST'])

    main_df = pd.merge(historical_dxgy_setups, drivers_context, how = 'left',
                       on = 'DRIVER_ID')
    main_df.columns = [x.lower() for x in main_df.columns]
    
    # define target (1 if reached the goal, 0 otherwise)
    main_df.loc[main_df['target_y'].isnull(), 'reached_goal'] = 0
    main_df.loc[~main_df['target_y'].isnull(), 'reached_goal'] = 1
        
    main_df['days_since_last_trip'] = (main_df['period_date_start'] - main_df['dt_last']).dt.days.clip(lower = 0)
    main_df['days_since_first_trip'] = (main_df['period_date_start'] - main_df['dt_first']).dt.days.clip(lower = 0)

    main_df.drop(['target_y','dt_last', 'dt_first'], axis = 1, inplace = True)
    main_df.fillna(0, inplace = True)

    return main_df

def get_valid_predictions(historical_dxgy_setups: pd.DataFrame,
                          clf_isotonic, clf_trips_isotonic, final_features_list: list,
                          optimal_spends_thresh: float, optimal_trips_thresh: float,
                          valid_start_date = '2021-07-02'):
    
    historical_dxgy_setups['PERIOD_DATE_START'] = pd.to_datetime(historical_dxgy_setups['PERIOD_DATE_START'])
    validation_setups = historical_dxgy_setups.loc[historical_dxgy_setups['PERIOD_DATE_START'] >= valid_start_date]\
                        .reset_index(drop = True)
    
    output = pd.DataFrame()
    error_setup_ids = []
    for setup in tqdm(validation_setups.ID.unique()):
        context_end_date = str(pd.to_datetime(validation_setups.loc[validation_setups['ID'] == setup, 'PERIOD_DATE_START'].values[0]))[:10]
        setup_info = validation_setups.loc[(validation_setups['ID'] == setup) & (validation_setups['PERIOD_DATE_START'] == context_end_date)]
        
        filtered_drivers_context = get_drivers_context(file_dir, sql_drivers_context_path,
                                                       locality, context_end_date)
        
        new_valid_df = prepare_valid_df(setup_info, filtered_drivers_context)
        new_valid_trips_df = new_valid_df.copy()
        fact_rides = setup_info['FACT_RIDES'].sum()
        fact_spends = setup_info['TARGET_Y'].sum()
        
        try:
            new_valid_df['calibrated_preds'] = clf_isotonic.predict_proba(new_valid_df[final_features_list])[:, -1]
            new_valid_df = new_valid_df.loc[new_valid_df['calibrated_preds'] > optimal_spends_thresh]
            spends_estimation = calc_expected_rides_and_spends(new_valid_df)
            spends_estimation = pd.DataFrame([spends_estimation])
            spends_estimation['model'] = 'spends'
            spends_estimation['setup_id'] = setup
            spends_estimation['fact_spends'] = fact_spends
            spends_estimation['fact_rides'] = fact_rides
            output = pd.concat([output, spends_estimation])
            del new_valid_df['calibrated_preds']
            new_valid_trips_df['calibrated_preds'] = clf_isotonic.predict_proba(new_valid_trips_df[final_features_list])[:, -1]
            new_valid_trips_df = new_valid_trips_df.loc[new_valid_trips_df['calibrated_preds'] > optimal_trips_thresh]
            trips_estimation = calc_expected_rides_and_spends(new_valid_trips_df)
            trips_estimation = pd.DataFrame([trips_estimation])
            trips_estimation['model'] = 'trips'
            trips_estimation['setup_id'] = setup
            trips_estimation['fact_spends'] = fact_spends
            trips_estimation['fact_rides'] = fact_rides
            output = pd.concat([output, trips_estimation], ignore_index = True)
            
        except ValueError:
            error_setup_ids.append(setup)
            pass
        
        return output

    

if __name__ == '__main__':
    
    # get and prepare data
    historical_dxgy_setups = get_dxgy_data(file_dir, sql_setups_query_path, locality)
    drivers_context = get_drivers_context(file_dir, sql_drivers_context_path,
                                          locality, end_date)
    main_df = prepare_main_df(historical_dxgy_setups, drivers_context)
    main_df = main_df.loc[main_df['period_date_start'] <= end_date].reset_index(drop = True)
    main_df = main_df.loc[main_df['trips'].notnull()].reset_index(drop = True)
    
    y, X = main_df['reached_goal'], main_df.drop(['reached_goal'] + unique_obs_groupper, axis = 1)
    
    # under sample to avoid overfitting
    rus = RandomUnderSampler(sampling_strategy = .5, random_state = 124)
    X_under, y_under = rus.fit_resample(X, y)

    # fit the classifier
    X_train, X_test, y_train, y_test = train_test_split(X_under, y_under,
                                                        random_state = 42,
                                                        train_size = .7)

    calibrated_rf = RandomForestClassifier(n_estimators = 500, 
                                           max_depth = 3,
                                           max_features = 3,
                                           random_state = 43)
    clf_isotonic = CalibratedClassifierCV(calibrated_rf, method = 'isotonic', cv = 5)
    clf_isotonic.fit(X_train[final_features_list], y_train)
    save_pickle(clf_isotonic, file_dir + '/data/calibrated_rf_scoring_reached_goals.pkl')
    
#    clf_isotonic = load_pickle(file_dir + '/data/calibrated_rf_scoring_reached_goals.pkl')

    # esimate trips and spends
    X_test['preds'] = clf_isotonic.predict(X_test[final_features_list])    
    print(f1_score(y_test, X_test['preds']))

    # validate on real setup
    driver_ids = read_file(file_dir + valid_driver_ids)
    valid_df = prepare_valid_set(driver_ids, sql_setups_query_path,
                                 sql_drivers_context_path)#, setup_id = 38667,
#                                 start_date = '2021-07-23', end_date = '2021-07-27')
    valid_df['calibrated_preds'] = clf_isotonic.predict_proba(valid_df[final_features_list])[:, -1]
    optimal_spends_thresh = get_optimal_threshold(y_train, clf_isotonic.predict_proba(X_train[final_features_list])[:, -1])     # optimal_spends_thresh = .43
    filtered_valid_df = valid_df.loc[valid_df['calibrated_preds'] > optimal_spends_thresh]
    print(calc_expected_rides_and_spends(valid_df))
    

"""
    Model for topline trips prediction (generated by activated drivers a.k.a at least 1 trip)
"""
    main_trips_df = prepare_main_df_for_trips_prediction(historical_dxgy_setups,
                                                         drivers_context)
    y_trips, X_trips = main_trips_df['activated'], main_trips_df[final_features_list]

    # udnersampling
    rus = RandomUnderSampler(sampling_strategy = .5, random_state = 123)
    X_under, y_under = rus.fit_resample(X_trips, y_trips)

    # fit the classifier
    X_trips_train, X_trips_test, y_trips_train, y_trips_test = train_test_split(X_trips, y_trips,
                                                                                random_state = 44,
                                                                                train_size = .7)

    calibrated_trips_rf = RandomForestClassifier(n_estimators = 500, 
                                                 max_depth = 3,
                                                 max_features = 3,
                                                 random_state = 45)
    clf_trips_isotonic = CalibratedClassifierCV(calibrated_trips_rf, method = 'isotonic', cv = 5)
    clf_trips_isotonic.fit(X_trips_train[final_features_list], y_trips_train)
    save_pickle(clf_trips_isotonic, file_dir + '/data/calibrated_rf_scoring_trips_topline.pkl')

#    clf_trips_isotonic = load_pickle(file_dir + '/data/calibrated_rf_scoring_trips_topline.pkl')

    # evaluate
    X_trips_test['preds'] = clf_isotonic.predict(X_trips_test[final_features_list])    
    print(f1_score(y_trips_test, X_trips_test['preds']))
    
    valid_trips_df = valid_df.copy()
    valid_trips_df['calibrated_preds'] = clf_trips_isotonic.predict_proba(valid_trips_df[final_features_list])[:, -1]
    optimal_trips_thresh = get_optimal_threshold(y_trips_train, clf_trips_isotonic.predict_proba(X_trips_train[final_features_list])[:, -1]) #   optimal_trips_thresh = .2
    filtered_valid_trips_df = valid_trips_df.loc[valid_trips_df['calibrated_preds'] > optimal_trips_thresh]

    # esimate trips and spends
    print(calc_expected_rides_and_spends(valid_trips_df))

    # FINAL EVALUATION
    evaluation_output = get_valid_predictions(historical_dxgy_setups, clf_isotonic,
                                              clf_trips_isotonic, final_features_list,
                                              optimal_spends_thresh, optimal_trips_thresh,
                                              valid_start_date = '2021-07-02')


"""
    END
"""


setups = {'target_x': [60, 66, 60, 66],
          'budget': [7000, 7000, 8000, 8000],
          'spends_model_preds': [],
          'trips_model_preds': []}

for target_x, budget in zip(setups['target_x'], setups['budget']):
    
    valid_df['target_x'] = target_x
    valid_df['budget'] = budget
    
    # spends model expectations
    valid_df['calibrated_spend_preds'] = clf_isotonic.predict_proba(valid_df[final_features_list])[:, -1]
    spends_model_preds = {'trips': valid_df['calibrated_spend_preds'].mean() * target_x * valid_df.driver_id.nunique(),
                          'spends':  valid_df['calibrated_spend_preds'].mean() * budget * valid_df.driver_id.nunique() * .84}
    setups['spends_model_preds'].append(spends_model_preds)

    # total trips model
    valid_df['calibrated_trips_preds'] = clf_trips_isotonic.predict_proba(valid_df[final_features_list])[:, -1]
    trips_model_preds = {'trips': valid_df['calibrated_trips_preds'].mean() * target_x * valid_df.driver_id.nunique(),
                         'spends':  valid_df['calibrated_trips_preds'].mean() * budget * valid_df.driver_id.nunique() * .84}
    setups['trips_model_preds'].append(trips_model_preds)
    valid_df.to_excel('~/Desktop/wokrflow/dxgy_mvp/data/results/dxgy_{0}_{1}.xlsx'.format(target_x, budget), index = False)
    
    