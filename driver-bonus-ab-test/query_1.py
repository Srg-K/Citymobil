q_1 = """
SELECT      BONUS_ID,
            SPLIT_NO,
            PERIOD,
            TO_DATE(PERIOD_DATE_START)                                                                       START_DT,
            TO_DATE(PERIOD_DATE_END)                                                                         END_DT,
            CASE
                WHEN TARGET2 IS NOT NULL AND TARGET1 <> TARGET2 THEN CONCAT(
                        COALESCE(TARGET1, PERIOD_ORDER_COUNT), '_', COALESCE(TARGET2, PERIOD_ORDER_COUNT))
                ELSE PERIOD_ORDER_COUNT
            END                                                                                          X,
            CASE
                WHEN TARGET2 IS NOT NULL AND TARGET1 <> TARGET2 THEN CONCAT(
                        COALESCE(PAYMENT1, ROUND(PAYMENT / 100)), '_', COALESCE(PAYMENT2, ROUND(PAYMENT / 100)))
                ELSE ROUND(PAYMENT / 100)
                END                                                                                          Y,
            DRIVER_ID,
            COUNT,
            PAID_SUM,
            ORDER_IDS
FROM (
         SELECT BONUS_ID
              , DRIVER_ID
              , SPLIT_NO
              , PAID_SUM
              , ORDER_IDS
              , PERIOD_DATE_START
              , PERIOD_DATE_END
              , PERIOD_ORDER_COUNT
              , COUNT
              , MAX(PAYMENT) PAYMENT
              , MIN(TARGETS)  TARGET1
              , MAX(TARGETS)  TARGET2
              , MIN(PAYMENTS) PAYMENT1
              , MAX(PAYMENTS) PAYMENT2
         FROM (
                             SELECT BONUS_ID
                                  , DRIVER_ID
                                  , PERIOD_DATE_START
                                  , PERIOD_DATE_END
                                  , PERIOD_ORDER_COUNT
                                  , PAYMENT
                                  , SPLIT_NO
                                  , PAID_SUM
                                  , ORDER_IDS
                                  , COUNT
                                  , UDF.JSON_TABLE(EXTRA_DATA, TRUE, '$.target_steps[*].target', '$.target_steps[*].payment')
                                                   EMITS
                                                   (TARGETS INT, PAYMENTS INT)
                             FROM REPLICA.DRIVER_BONUS_ORDER_COUNTS
                             WHERE PERIOD_DATE_START >= '2020-03-01'
                              AND BONUS_ID IN ('{0}') 
               )
         GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
         ) DBO
         LEFT JOIN REPLICA.DRIVER_BONUS DB ON DBO.BONUS_ID = DB.ID
         LEFT JOIN MD.LOCALITY L ON ID_LOCALITY = L.LOCALITY_RK
"""