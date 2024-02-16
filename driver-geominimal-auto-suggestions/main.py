__author__ = "Igor Singin"
__version__ = "2022-01-10"
__email__ = "i.singin@city-mobil.ru"
__status__ = "Prototype"

import logging
import pandas as pd
import utils
import queries
from configparser import ConfigParser
import ast
import db
import sys

def main():
    config = ConfigParser()
    config.read("config.ini")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()

    logger.info('Service start working...')
    try:
        mysql           = db.get_mysql_conn()
        clickhouse_main = db.get_ch_main_conn()
    except Exception as err:
        logger.error(f"No DWH connection: {err}, {type(err)}")
        exit(1)

    ### Словарь значений лимитов для каждого города по умолчанию
    city_limits = ast.literal_eval( config['cities']['city_limits'] )

    ### Название столбца где хранится id кластера
    labels             = config['clustering']['labels']
    print_info_flag    = bool(config['printing_logging']['print_info_flag'])

    ### Выгрузка данных запроса таксисерва из DWH. Все проверки внутри.
    df_request = utils.get_request_from_taxiserv(mysql,logger)

    locality              = int(df_request['locality_id'])
    limit_sum             = int(df_request['limit_sum'])
    request_datetime      = df_request['created_dt'][0].to_pydatetime()
    taxiserv_username     = int(df_request['created_by'][0])
    cnt_days_for_download = int(df_request['train_period'])
    id_request            = int(df_request['id'])
    status                = str(df_request['status'])
    calculation_date      = str(request_datetime.date())

    ### Находим список ограничений для города из словаря с городами по locality
    city_limits_list = city_limits.get(str(locality))
    city_limits_list[0] = limit_sum
    median_bill_multiplicator = city_limits_list[5]

    logger.info(f'Start geominimal calculation for city={locality} date = {calculation_date} cnt_days = {cnt_days_for_download} id_request={id_request}')
    ###  Сообщаем о начале рассчета и записываем в таблицу
    utils.update_request_table_start(id_request, 'IN_PROGRESS', mysql ,logger)

    ### Выгружаем данные из Кликхауса
    data = utils.download_data(id_request
                             , locality
                             , calculation_date
                             , cnt_days_for_download
                             , queries.query_d2v_from_surge
                             , queries.query_filtered_client_bill
                             , queries.query_info_into_gexagon
                             , clickhouse_main
                             , mysql
                             , logger
                             )

    ### Проверка существования и корректности используемых данных
    utils.check_dataframe_health(data
                                , id_request
                                , ['weekday', 'hour', 'hex_id', 'geo_lat', 'geo_long', 'fact_orders', 'fact_rides']
                                , 'Problem with "data" dataframe before data_quality_validation'
                                , mysql
                                , logger  )
    flag_data_quality = utils.data_quality_validation(data
                                                , calculation_date
                                                , cnt_days_for_download
                                                , locality
                                                , queries.query_check_problems_with_data_source
                                                , clickhouse_main
                                                , logger      )

    ### НАХОДИМ фильтры для попадания в бюджет
    utils.check_dataframe_health(data
                                , id_request
                                , ['weekday', 'hour', 'hex_id', 'main_id_locality', 'geo_lat', 'geo_long', 'fact_orders', 'fact_rides', 'filtered_avg_driver_bill']
                                , 'Problem with "data" dataframe before find_budget_treshold_for_locality'
                                , mysql
                                , logger )
    filter1, filter2 = utils.find_budget_treshold_for_locality(data, locality, city_limits_list, logger)

    ### Фильтруем гексагоны используя найденные фильтры
    utils.check_dataframe_health(data
                                , id_request
                                , ['weekday', 'main_id_locality']
                                , 'Problem with "data" dataframe before get_filtered_data_daily_tr'
                                , mysql
                                , logger )
    data_filtered = utils.get_filtered_data_daily_tr(data
                                               , ['main_id_locality', 'weekday']
                                               , filter2 #quantile_views
                                               , filter1 #quantile_d2v
                                               )

    logger.info(f'Final run, use filter1={filter1} and filter2 = {filter2}')
    ### Находим кластеры и считаем геоминималки в кластерах
    geominimals_locality = utils.incentives_evaluation(data_filtered, locality, median_bill_multiplicator, logger)

    ### Формируем выходной датасет
    if (geominimals_locality[geominimals_locality.geominimal > 0].shape[0] > 0):
        geominimals_locality = geominimals_locality[geominimals_locality.geominimal > 0]
        result_locality = utils.create_result_dataset(geominimals_locality, logger)

        ### Если нет проблем с качеством данных сохраняем результаты
        if (flag_data_quality == 'ok'):
            result = pd.DataFrame()
            result = result.append(result_locality)
            result['locality_id'] = locality
            result['request_id'] = id_request
            utils.print_setup_results(result, locality, logger)
            mysql.insert_data_frame('geominimal_suggest_data', result)
            utils.update_request_table(id_request, 'READY', mysql, logger)
        ### Если есть проблемы с качеством данных не сохраняем результаты
        else:
            logger.error('Problem with data quality! No saved setup!')
            utils.update_request_table(id_request, 'FAILED', mysql, logger)


if __name__ == '__main__':
    main()
else:
    raise Exception("You can't import main.py, you have to run it directly with 'python main.py'")
