def q_1():
    # Активные эксперименты ГД
    return f'''SELECT *
    FROM driver_surcharge_list
    WHERE TRUE
    AND active = 1
    AND id in (7411)
    AND JSON_LENGTH(stairs) > 1
    '''


def q_2(sur_id):
    # Оплаты
    return f'''SELECT id
    , date
    , surcharge_id
    , driver_id
    , orders_count
    , driver_bill
    , price
    , expected_sum
    , fee
    , status
    , payment_id
    FROM driver_surcharge_payments 
    WHERE TRUE
    AND surcharge_id = {sur_id}'''


def q_3(sur_id, date):
    # Сплиты водителей
    return f'''select driver_id, date, split
    from driver_surcharge_participants
    where surcharge_id = {sur_id}
    and date = \'{date}\' '''
