que_1 = """
SELECT
    toDate('{0}')     as rep_date,
    d.locality_id        as region_id,
    d.id                 as driver_id,
    oc.recency           as recency,
    oc.frequency         as frequency,
    p.monetary           as monetary
FROM
(
   select id, locality_id
   from segmentation.drivers d FINAL
   where d.first_order_date < toDateTime(toDate('{0}') - toIntervalWeek(4))
   and d.locality_id in {1}
   and d.id not in
      (10367166, 10024714, 10182218, 10166470, 144646, 10669938, 10706746,
       10122334, 10552990, 10467046, 78)
   and d.id not in
      (10155434, 29928, 5883, 10022482, 23948, 1241, 14482, 10584286,
       4466, 3970, 4915, 8829, 13924,
       37174, 19544, 31393, 1024, 2563, 1953, 2372, 18916, 10123, 3587,
       4843, 4793, 2603, 3024,
       2055, 137310, 2689, 10223042, 2582, 6629, 5091, 5883, 6873, 1933,
       847, 2012, 2083, 9487,
       8981, 10004978, 400, 27378, 28744, 10222402, 1001, 21398, 33294,
       12932, 6706, 23362,
       10151, 15674, 153, 7367, 18952,
       20178, 4646, 10335754, 14110,
       19266, 19448, 2592, 1049, 4351,
       3747, 15202, 1206, 13584, 13718,
       10218866, 13168, 104812, 5580,
       5257, 5842, 33944, 33496, 10515202,
       6149, 29862, 748, 3535, 12846,
       8351, 26902, 3409, 714, 10608914,
       5642, 10531542, 4181, 23852, 1417,
       10207230, 20334, 20408, 1878,
       14866, 147500, 14168, 10224062,
       3477, 10405254, 7293, 30330, 30238, 29254,
       108966, 10718402, 10582690, 14250,
       16204, 15610, 18398, 16184, 17002,
       32248, 5298, 135280, 122150, 6953,
       32282, 4624, 20686, 4216, 4204,
       15870, 5901, 36970, 3692, 10718886,
       154838, 10155434, 139474, 21380, 1371, 18952, 3122,
       24896, 125556, 2922, 4467,
       10011942, 13850, 157992, 29902, 160606, 9994642, 10055038,
       10050786, 119198, 132622, 153634, 155660, 29538,
       101224, 121222, 172568, 10050806, 149106, 9992456, 103034, 126428,
       169056, 10034318,
       6979, 10344610, 23370, 9493, 26900, 6979, 10344610, 6952, 2571,
       10603826, 11301434, 12882434)
   and d.is_test != 1
   and d.is_active = 1
) d
GLOBAL ALL
INNER JOIN
(
   select driver_id, groupArray(oc.id) ids,
          count(DISTINCT (toMonday(toDate(oc.created_date)))) AS weeks,
          count(DISTINCT oc.id)                               AS cnt,
          toDate('{0}') - toDate(max(oc.created_date)) AS recency,
          round(cnt / weeks)                                  AS frequency
   from segmentation.orders_dist oc FINAL
   where created_date >= toDateTime(toDate('{0}') - toIntervalWeek(4))
     and created_date < toDateTime(toDate('{0}'))
     and oc.status = 'CP'
     and oc.is_test != 1
   group by driver_id
) oc
ON toUInt64(d.id) = oc.driver_id
GLOBAL ALL
INNER JOIN
(
    SELECT driver_id, (SUM(driver_payment_monetary) + SUM(company_payment_monetary)) monetary
    FROM (
        select driver_id, id, sum, round(if(company_id = 1, -1 * sum, 0) / 100) as driver_payment_monetary
        from segmentation.driver_payments_dist FINAL
        where type_id in (1, 566, 774)
          and sum < 0
          and time >= toDateTime(toDate('{0}') - toIntervalWeek(4))
          and time < toDateTime(toDate('{0}'))
    ) dp
             GLOBAL ALL
             LEFT JOIN
         (
             SELECT round(debit / 100) company_payment_monetary, driver_payment_id
             FROM segmentation.company_payments_dist
             where type_id in (2, 322, 222)
               and time >= toDateTime(toDate('{0}') - toIntervalWeek(4))
               and time < toDateTime(toDate('{0}'))
             ) cp
         ON cp.driver_payment_id = dp.id
    GROUP BY driver_id
) p
ON p.driver_id = toUInt64(d.id)
"""
