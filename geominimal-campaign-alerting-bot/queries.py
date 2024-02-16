order_metadata_daily_sql = '''
    with sk as (
        select opc.ID_ORDER_FINAL, max(opc.COEFFICIENT) as coef
        from replica.ORDER_PRICE_CALCULATIONS opc
        where opc.G_TYPE is null /*исключаем тестовые заказы*/
        and to_date(PRICE_TIME) = current_date -1
        group by opc.ID_ORDER_FINAL)
    select o.ORDER_RK, o.LOCAL_ORDER_DTTM,
    case when minute(o.LOCAL_ORDER_DTTM) < 30
        then TO_TIMESTAMP(concat(to_date(o.LOCAL_ORDER_DTTM),' ', hour(o.LOCAL_ORDER_DTTM),':00:00'))
        else TO_TIMESTAMP(concat(to_date(o.LOCAL_ORDER_DTTM),' ', hour(o.LOCAL_ORDER_DTTM),':30:00'))
        end slot,
    o.LOCALITY_RK, o.LOCALITY_NM,
        sk.coef / 10 surge_coef
    from EMART."ORDER" o
    LEFT JOIN sk on sk.ID_ORDER_FINAL = o.order_rk
    where STATUS_CD='CP'
        and to_date(o.LOCAL_ORDER_DTTM) = current_date -1
        and o.DRIVER_RK is not null
'''

geominimal_stat_daily_sql = '''
    SELECT order_id ORDER_RK, driver_id DRIVER_RK, geo_minimal_price GEOMINIMAL_PRICE,
        driver_bill DRIVER_BILL_AMT, driver_bill - base_price DI_GEO_MINIMAL_AMT,
           geo_minimal_schedule_id, geo_minimal_polygon_id
    from geominimal.geominimal_stat gs final
    where toDate(gs.order_created) = today()-1
        and base_price > 0
        and minimal_source_name = 'geominimal'
        and geo_minimal_enough_points = 1
        and geo_minimal_is_hit_to_split = 1
'''

gch_daily_sql = '''
    select gch.ID, gch.ID_LOCALITY , gch.CAMPAIGN_ID, gch.CAMPAIGN_NAME,
           JSON_EXTRACT(gch.POLYGONS,
                   '$#.id',
                   '$#.name')
                   EMITS (
                       polygon_id varchar(1000),
                       polygon_name varchar(100))
    from replica.geominimal_change_history gch
    where gch.POLYGONS != '[]'
        and (gch.TIME_END is null or to_date(gch.TIME_END) >= current_date-1)
'''


order_metadata_weekly_sql = '''
    with sk as (
        select opc.ID_ORDER_FINAL, max(opc.COEFFICIENT) as coef
        from replica.ORDER_PRICE_CALCULATIONS opc
        where opc.G_TYPE is null /*исключаем тестовые заказы*/
        and to_date(PRICE_TIME)
            between DATE_TRUNC('WEEK', current_date) - 7 and DATE_TRUNC('WEEK', current_date) - 1
        group by opc.ID_ORDER_FINAL)
    select o.ORDER_RK, o.LOCAL_ORDER_DTTM,
    case when minute(o.LOCAL_ORDER_DTTM) < 30
        then TO_TIMESTAMP(concat(to_date(o.LOCAL_ORDER_DTTM),' ', hour(o.LOCAL_ORDER_DTTM),':00:00'))
        else TO_TIMESTAMP(concat(to_date(o.LOCAL_ORDER_DTTM),' ', hour(o.LOCAL_ORDER_DTTM),':30:00'))
        end slot,
    to_char(o.LOCAL_ORDER_DTTM, 'd') week_day_dt,
    o.LOCALITY_RK, o.LOCALITY_NM,
        sk.coef / 10 surge_coef
    from EMART."ORDER" o
    LEFT JOIN sk on sk.ID_ORDER_FINAL = o.order_rk
    where STATUS_CD='CP'
        and to_date(o.LOCAL_ORDER_DTTM)
            between DATE_TRUNC('WEEK', current_date) - 7 and DATE_TRUNC('WEEK', current_date) - 1
        and o.DRIVER_RK is not null
'''

geominimal_stat_weekly_sql = '''
    SELECT order_id ORDER_RK, driver_id DRIVER_RK, geo_minimal_price GEOMINIMAL_PRICE,
        driver_bill DRIVER_BILL_AMT, driver_bill - base_price DI_GEO_MINIMAL_AMT,
           geo_minimal_schedule_id, geo_minimal_polygon_id
    from geominimal.geominimal_stat gs final
    where toDate(gs.order_created) between toStartOfWeek(today(), 1) - 7 and toStartOfWeek(today(), 1)- 1
        and base_price > 0
        and minimal_source_name = 'geominimal'
        and geo_minimal_enough_points = 1
        and geo_minimal_is_hit_to_split = 1
'''

gch_weekly_sql = '''
    select gch.ID, gch.ID_LOCALITY , gch.CAMPAIGN_ID, gch.CAMPAIGN_NAME,
           JSON_EXTRACT(gch.POLYGONS,
                   '$#.id',
                   '$#.name')
                   EMITS (
                       polygon_id varchar(1000),
                       polygon_name varchar(100))
    from replica.geominimal_change_history gch
    where gch.POLYGONS != '[]'
        and (gch.TIME_END is null or to_date(gch.TIME_END) >= DATE_TRUNC('WEEK', current_date) - 7)
'''
