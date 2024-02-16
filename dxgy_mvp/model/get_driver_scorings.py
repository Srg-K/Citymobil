#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:36:15 2021

@author: skhalilbekov
"""
import os
import pyexasol
import pandas as pd

from utils.utils import load_pickle, sql_from_file


def get_drivers_context(file_dir: str, sql_drivers_context_path: str, locality: int, 
                        end_date: str, driver_ids: pd.DataFrame, driver_id_col: str):
    
    query = sql_from_file(file_dir + sql_drivers_context_path)\
                    .format(locality, end_date)
    
    cred = pd.read_json(r'/Users/skostuchik/crd_exa.json')
    user = cred.iloc[0, 0]
    password = cred.iloc[0, 1]
    db_host = pyexasol.connect(dsn='ex1..3.city-srv.ru:8563', user=user, password=password, fetch_dict=True)

    df = db_host.export_to_pandas(query)
    df = df.loc[df.DRIVER_ID.isin(driver_ids[driver_id_col].unique())]

    return df

def prepare_valid_df(drivers_context: pd.DataFrame, target_x: int, budget: int,
                     period: int, period_date_start: str):
    
    # preprocess
    main_df = drivers_context.copy()
    main_df.columns = [x.lower() for x in main_df.columns]
    main_df['dt_first'] = pd.to_datetime(main_df['dt_first'])
    main_df['dt_last'] = pd.to_datetime(main_df['dt_last'])
    main_df['period_date_start'] = pd.to_datetime(period_date_start)
    main_df['period_date_start'] = pd.to_datetime(main_df['period_date_start'])
    
    # define features
    main_df['days_since_last_trip'] = (main_df['period_date_start'] - main_df['dt_last']).dt.days.clip(lower = 0)
    main_df['days_since_first_trip'] = (main_df['period_date_start'] - main_df['dt_first']).dt.days.clip(lower = 0)
    main_df['target_x'] = target_x
    main_df['budget'] = budget
    main_df['period'] = period
    
    main_df.drop(['period_date_start','dt_last', 'dt_first'], axis = 1, inplace = True)
    if main_df.isnull().sum().sum() > 0:
        main_df.fillna(0, inplace = True)
        print('===== There NaN in df - filled with 0')
        
    return main_df

def load_dxgy_models(file_dir: str):
    
    output = {'spends_model': None,
              'trips_model': None}
        
    output['spends_model'] = load_pickle(file_dir + '/data/calibrated_rf_scoring_reached_goals.pkl')    
    output['trips_model'] = load_pickle(file_dir + '/data/calibrated_rf_scoring_trips_topline.pkl')
    
    return output

def get_dxgy_estimations(drivers_context: pd.DataFrame, models: dict, file_dir: str,
                         target_x: list, budget: list, period: list,
                         period_date_start: list, final_features_list: list):
    
    setups = {'target_x': target_x,
              'budget': budget,
              'period': period,
              'period_date_start': period_date_start,
              'spends': [],
              'trips': []}
    
    for target_x, budget, period, period_date_start in zip(setups['target_x'], setups['budget'], \
                                                           setups['period'], setups['period_date_start']):
        
        # preprocess and calc features
        main_df = prepare_valid_df(drivers_context, target_x, budget,
                                   period, period_date_start)
        
        # spends model expectations
        main_df['calibrated_spend_preds'] = models['spends_model'].predict_proba(main_df[final_features_list])[:, -1]
        spends_model_preds = main_df['calibrated_spend_preds'].mean() * budget * main_df.driver_id.nunique() * .84
        setups['spends'].append(spends_model_preds)
    
        # total trips model
        main_df['calibrated_trips_preds'] = models['trips_model'].predict_proba(main_df[final_features_list])[:, -1]
        trips_model_preds = main_df['calibrated_trips_preds'].mean() * target_x * main_df.driver_id.nunique()
        setups['trips'].append(trips_model_preds)

        # save each preds for double-check
        main_df.to_excel(f'{file_dir}/data/results/dxgy_{0}_{1}.xlsx'.format(target_x, budget), index = False)
        
        return pd.DataFrame(setups)