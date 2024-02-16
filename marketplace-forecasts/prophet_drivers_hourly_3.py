import logging
import pandas as pd
import numpy as np
import db
import warnings
warnings.simplefilter('ignore')

from datetime import datetime
from fbprophet import Prophet
import holidays
from sql_load_data import que_drivers, que_weather, que_weather_forecast
from config import city_config


dt = datetime.now()
dt0 = dt.strftime("%Y-%m-%d %H")

logger = logging.getLogger()
mysql = db.get_mysql_conn()
exasol = db.get_exasol_connection()
clickhouse = db.get_ch_seg_conn()

holidays_dict = holidays.RU(years=(2019, 2020, 2021))
df_holidays = pd.DataFrame.from_dict(holidays_dict, orient='index') \
    .reset_index()
df_holidays = df_holidays.rename({'index':'ds', 0:'holiday'}, axis ='columns')
df_holidays['ds'] = pd.to_datetime(df_holidays.ds)
df_holidays = df_holidays.sort_values(by=['ds'])
df_holidays = df_holidays.reset_index(drop=True)

for city in ['saint_p','moscow']:
    df = pd.DataFrame(exasol.execute(que_drivers.format(city_config[str(city)+'_locality'],city_config[str(city)+'_tarif'])))
    df.columns = ['ds', 'y']
    df['ds']= pd.to_datetime(df['ds'])
    df = df.drop(df.index[-1:])

    weather = pd.DataFrame(clickhouse.fetch(que_weather.format(city_config[str(city)+'_locality'])))
    weather_forc = pd.DataFrame(clickhouse.fetch(que_weather_forecast.format(city_config[str(city)+'_locality'])))
    df = df.merge(weather, how='left', left_on='ds', right_on='dt')
    # df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
    df['temperature'] = df['temperature'].fillna(df['temperature'].median())
    df['temperature_human'] = df['temperature_human'].fillna(df['temperature_human'].median())
    df['precipitation_volume'] = df['precipitation_volume'].fillna(0)

    m = Prophet(
        yearly_seasonality=20,
        weekly_seasonality=20,
        daily_seasonality=False,
        holidays=df_holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=20,
        seasonality_mode='multiplicative',
        interval_width=0.95
    ).add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=12
    ).add_seasonality(
        name="daily",
        period=1,
        fourier_order=35
    ).add_regressor(
        'temperature'
    ).add_regressor(
        'temperature_human'
    ).add_regressor(
        'precipitation_volume'
    ).fit(df)
    future = m.make_future_dataframe(periods=49, freq='H')
    future = future.merge(weather_forc, how='inner', left_on='ds', right_on='dt')
    fcst = m.predict(future)

    target_cols = ['yhat', 'yhat_lower', 'yhat_upper']
    forc = fcst.loc[:, target_cols]
    forc[forc < 0] = 0
    forc['ds'] = fcst['ds']

    forecast = forc[forc['ds'] >= dt0]
    forecast[target_cols] = forecast[target_cols].round(0).astype('int')
    forecast['created_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    forecast['version'] = "prophet_03"
    forecast['locality_id'] = city_config[str(city)+'_locality']
    result = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'created_date', 'version','locality_id']]
    result.columns = ['date_hour', 'drivers', 'drivers_lower', 'drivers_upper', 'created_date', 'version','locality_id']
    mysql.insert_data_frame("forecast_drivers_hourly", result)
