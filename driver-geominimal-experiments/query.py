import config


def q_1():
    """
    SCHEDULE_DF
    Активные геоминималки в тесте

    Returns
    -------

    """
    return """
    SELECT campaign_id
       , id_locality
       , active as status
       , DATE(campaign_from) as from_date
       , DATE(campaign_until) as until_date
       , week_day
       , from_hour
       , from_minute
       , until_hour
       , until_minute
       , polygons
       FROM  geominimal_change_history
       WHERE 1 = 1
           and percent_drivers < 100
           and active = 1
           and time_end is null
           and campaign_from <= current_date
           and campaign_until >  DATE(DATE(current_date)-1)
    """


def q_2(campaign_id):
    """
    POLYGON_DF
    Справочник полигонов

    Returns
    -------

    """
    return f"""
    SELECT polygon_id, campaign_id, id, id_locality, coords
        FROM city.polygon_to_geominimal_campaign cp
        LEFT JOIN polygons p ON (p.id = cp.polygon_id)
        WHERE cp.campaign_id = {campaign_id};
    """


def q_2_new_format(campaign_id):
    return f"""
    select distinct *
    from(
        SELECT campaign_id
             , id_locality
             , JSON_EXTRACT(polygons_flatten, '$.id', '$.coords.json()', '$.error()'
          ) EMITS (
                   id INT,
                   coords VARCHAR(2000000),
                   error_column VARCHAR(2000000))
        FROM (
              SELECT campaign_id
                   , id_locality
                   , JSON_EXTRACT(polygons, '$#.json()', '$.error()'
                ) EMITS (
                         polygons_flatten VARCHAR(2000000),
                         error_column VARCHAR(2000000)
                                    ) -- массив coords полигонов
              FROM REPLICA.GEOMINIMAL_CHANGE_HISTORY
              WHERE 1 = 1
                and CAMPAIGN_ID = {campaign_id}
               )
    )
    """


def q_3(campaign_id):
    """
    GEOMSTAT_DF
    Статистика по геоминимальным поездкам

    Returns
    :param campaign_id:
    :return:
    """
    return f'''SELECT order_id
        , driver_id
        , campaign_id
        , driver_bill - driver_price_without_geominimal as damage
        , 1 as is_geominimal_trip
        FROM city.geominimal_stat
        WHERE campaign_id =  {campaign_id}'''


def q_3_new_format(campaign_id):
    return f'''
        SELECT order_id
            , driver_id
            , geo_minimal_campaign_id campaign_id
            , if(base_price < geo_minimal_price and geo_minimal_enough_points = 1 and geo_minimal_is_hit_to_split = 1,  geo_minimal_price - base_price, 0) as damage
            , 1 as is_geominimal_trip
            FROM  geominimal.geominimal_stat
        WHERE geo_minimal_campaign_id =  {campaign_id}
    '''


def q_4(campaign_id, locality):
    """
    Сплиты водителей

    :param campaign_id:
    :param locality:
    :return:
    """
    return f'''SELECT id, CRC32(CONCAT({campaign_id}, '_', id, '_jib9')) % 100 < (SELECT percent_drivers FROM geominimal_campaign WHERE id={campaign_id}) split
            FROM drivers
            WHERE id_locality = {locality} AND role=11 '''


def q_5(polygons, locality, timings, from_date, until_date, period):
    """

    :param polygons:
    :return:
    """
    if period == 'lw':
        condition = f"""\'{from_date}\' - 7 AND DATE \'{until_date}\' - 6"""
    elif period == 'tw':
        condition = f"""\'{from_date}\' AND DATE \'{until_date}\' +1"""

    q = f"""
    with p as (select 'POLYGON '||replace(replace(replace(replace(COORDS, '","', ' '), '"],["', ','), '[["', '(('), '"]]','))') as coord
                 from REPLICA.POLYGONS
                 where id in ({",".join(polygons)})
                 order by false)
            , t as
            ( select oc.ID
                , fss.ID fss_id
                , ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET) as DATE_SUGGEST1
                , oc.ID_CLIENT
                , fss.ID_DRIVER
                , oc.DRIVER_BILL
                , oc.ID_POLYGON
                , oc.STATUS
                , oc.DRIVER
                , fss.ACTION
                , fss.DRIVER_LAT
                , fss.DRIVER_LON
                , sh.yellow_idle_next_order_sec + sh.home_idle_sec + sh.green_idle_sec + sh.on_trip_sec+ sh.to_client_collected_sec SUPPLY_SEC
                , coord
                , oc.g_type
                , \'{period}\' as retro
            from replica.order_closed oc
                left join md.locality l on l.locality_rk = oc.main_id_locality
                left join replica.fairbot_suggests_success fss on oc.id = fss.id_order
                left join replica_mart.drivers_sh_orderly sh on oc.id = sh.order_rk
                inner join p on st_intersects(p.coord, 'POINT ('||fss.driver_lat||' '||fss.driver_lon||')')
            WHERE ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET) BETWEEN DATE {condition}
                AND ({timings})
                AND oc.MAIN_ID_LOCALITY = {locality}
                AND oc.get_company <> 3)
    
            select *
            from t
            where coord is not null
    """
    if locality != 22551:
        q += ' AND g_type = 0'
    print(q)

    return q


def q_5_new_format(locality, timings, from_date, until_date, period):
    """

    :param polygons:
    :return:
    """
    if period == 'lw':
        condition = f"""\'{from_date}\' - 7 AND DATE \'{until_date}\' - 6"""
    elif period == 'tw':
        condition = f"""\'{from_date}\' AND DATE \'{until_date}\' +1"""

    q = f"""
    select * 
    from (
        select oc.ID
            , fss.ID fss_id
            , ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET) as DATE_SUGGEST1
             , TO_CHAR(ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET), 'D') week_day
             , hour(ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET)) h
             , minute (ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET)) m
            , oc.ID_CLIENT
            , fss.ID_DRIVER
            , oc.DRIVER_BILL
            , oc.ID_POLYGON
            , oc.STATUS
            , oc.DRIVER
            , fss.ACTION
            , fss.DRIVER_LAT
            , fss.DRIVER_LON
            , sh.yellow_idle_next_order_sec + sh.home_idle_sec + sh.green_idle_sec + sh.on_trip_sec+ sh.to_client_collected_sec SUPPLY_SEC
            , oc.g_type
            , \'{period}\' as retro
            -- , sum(1) over(partition by oc.id, fss.ID_DRIVER order by fss.DATE_SUGGEST desc) as suggest_num
        from replica.order_closed oc
            left join md.locality l on l.locality_rk = oc.main_id_locality
            left join replica.fairbot_suggests_success fss on oc.id = fss.id_order
            left join replica_mart.drivers_sh_orderly sh on oc.id = sh.order_rk
        WHERE ADD_SECONDS(fss.DATE_SUGGEST,TIME_OFFSET) BETWEEN DATE {condition}
            AND ({timings})
            AND oc.MAIN_ID_LOCALITY = {locality}
            AND oc.get_company <> 3
    )
    where 1=1
    -- and suggest_num = 1
     
    """
    if locality != 22551:
        q += ' AND g_type = 0'

    return q