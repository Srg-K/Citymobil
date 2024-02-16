import config


def q_1(ci_id: int, dt_start, dt_end):
    """
    Данные КХ

    Returns
    -------
    """
    q =  f"""
    SELECT
          id_locality
        , id_tariff_zone
        , id_tariff_group
        , polygon_from
        , h3_6
        , dt
        , arrayJoin(momental_group) momental_group
        , ci_id
        , status
        , id_client
        , uniqExact(id_client)                                               views
        , uniqExactIf(id_order, id_order > 0)                                orders
        , uniqExactIf(id_order, id_order > 0 AND status NOT IN ('CC', 'NC')) rides
        , sumIf(client_bill, status = 'CP')                                  client_bill
        , sumIf(sale_part, sale_part > 0 AND status = 'CP')                  sale_part_ride
    FROM (
         SELECT id_order
              , id_locality
              , id_tariff_zone
              , id_tariff_group
              , polygon_from
              , h3_6
              , arrayJoin(client_impacts_ids)                                 ci_id
              , client_impacts_test_groups as momental_group
              , toStartOfInterval(calculated_dttm, interval 20 minute) dt
              , id_client
              , arraySum(discounts_absolute)                                  sale_part
              , status
              , client_bill
         FROM (
                SELECT uuid
                     , id_client
                     , discounts_absolute
                     , calculated_dttm
                     , client_impacts_ids
                     , client_impacts_test_groups
                     , id_locality
                     , id_tariff_zone
                     , id_tariff_group
                     , polygon_from
                     , h3ToParent(from_h3_15,6) h3_6
                FROM ods_fares.order_price_calculations
                WHERE calculated_dttm > \'{dt_start}\'
                  AND collected_dttm < \'{dt_end}\'
                  AND is_test = 0
                  AND has(client_impacts_ids,{ci_id})
                ) opc
                LEFT JOIN
              (
                SELECT idhash_uuid
                     , id_order
                     , status
                     , client_bill
                FROM ods_fares.order_status_info
                PREWHERE id_order > 0
                WHERE (OrderedDate > \'{dt_start}\' and OrderedDate < \'{dt_end}\')) oc
              ON opc.uuid = oc.idhash_uuid)
    GROUP BY   id_locality, id_tariff_zone, id_tariff_group, polygon_from, dt
          , momental_group, ci_id, id_client, status, h3_6
    """
    # print(q)
    return q


def q_2():
    """
    Тариффные группы (mysql 50.85)

    Returns
    -------

    """
    return """
    select id as id_tariff_group, name as tariff_group_name
    from city.tariff_groups
    """


def q_3():
    """
    Локаль (mysql 50.85)

    Returns
    -------

    """
    return """
    select id as id_locality, short_name as locality_name
    from city.google_locality
    """


def q_4():
    """
    Тариффные зоны (mysql 50.85)

    Returns
    -------

    """
    return """
    select id id_tariff_zone, name as tariff_zone_name
    from city.tariff_zones
    """


def q_5():
    """
    Таксопарки / партнеры (ch)

    Returns
    -------

    """
    return """
    select toUInt16(id) as company_id, name
    from rt_city_import.company
    """


def q_6():
    """
    Полигоны (mysql 50.85)

    Returns
    -------

    """
    return """
    SELECT id AS polygon_from, name AS polygon_name
    FROM city.polygons
    """


def q_7():
    """
    Client Impacts (51.205)

    Returns
    -------
    """
    return f"""
    SELECT id ci_id, date_started, date_ended
    FROM city3.client_impacts ci
    JOIN ( #только активные акции
        SELECT distinct client_impact_id
        FROM city3.client_impacts_parameters_history
        where parameter_name = 'activity'
            and parameter_value = 1
            and date_deleted is null
        ) a on a.client_impact_id = ci.id
    WHERE true
        and date_started < current_date
        and (date_ended > current_date or date_ended is null)
        and date_started > \'{config.DT_START_TRESHOLD}\'
    """
