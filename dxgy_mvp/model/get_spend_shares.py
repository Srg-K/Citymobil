#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:56:39 2021

@author: skhalilbekov
"""

import pandas as pd

from utils.utils import *


def get_spends_data(locality: int, file_dir: str, spends_query_path: str):
    
    query = sql_from_file(f'{file_dir}/{spends_query_path}').format(locality)

    db_host = pyexasol.connect(dsn = 'ex1..3.city-srv.ru:8563',
                              user = os.getenv('MAIN_USER'),
                              password = os.getenv('PASSWORD'), 
                              fetch_dict = True)

    spends_df = db_host.export_to_pandas(query)
        
    return spends_df

def get_spend_shares(locality: int, file_dir: str,
                     spends_query_path: str, resample_type = 'M'):
    
    spends_df = get_spends_data(locality, file_dir, spends_query_path)
    
    spends_df['DT'] = pd.to_datetime(spends_df['DT'])   
    spends_df = spends_df.set_index('DT')
        
    # calc shares
    new_cols = []
    groupped_df = spends_df.resample('M').sum().sort_index()
    for col in ['DXGY_TRIPS', 'MFG_TRIPS', 'GEOMINIMAL_TRIPS', 'DXGY_MFG_TRIPS', 'DXGY_GEOMINIMAL_TRIPS', 'MFG_GEOMINIMAL_TRIPS', 'ALL_IC_TRIPS']:
        groupped_df[f'{col}_share'] = groupped_df[f'{col}'] / groupped_df['TRIPS']    
        new_cols.append(f'{col}_share')
"""        
    for col in new_cols:
        groupped_df[col] = groupped_df[col] / groupped_df[new_cols].sum(axis = 1)
"""    
    
    