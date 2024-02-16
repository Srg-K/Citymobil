import db
import notifination
import queries as q
import responsibles as r

import pandas as pd


order_metadata = pd.DataFrame(db.get_exasol_connection().execute(q.order_metadata_daily_sql).fetchall())
gch = pd.DataFrame(db.get_exasol_connection().execute(q.gch_daily_sql).fetchall())

order_metadata['SURGE_COEF'] = order_metadata['SURGE_COEF'].astype(float)
gch['POLYGON_ID'] = gch['POLYGON_ID'].astype(int)

geominimal_stat = db.get_ch_seg_conn().fetch(q.geominimal_stat_daily_sql)

gh_data = geominimal_stat.merge(order_metadata, how='inner', on='ORDER_RK')
gh_data = gh_data.merge(gch, how='left', left_on=['geo_minimal_schedule_id','geo_minimal_polygon_id'],
                        right_on=['ID','POLYGON_ID'])
gh_data['SURGE_COEF'] = gh_data['SURGE_COEF'].fillna(1)


if len(gh_data) == 0:
    notifination.send_to_slack(":harold2: В данный момент есть проблемы с данными, скоро все заработает :harold2:")
else:
    df = gh_data.groupby(['LOCALITY_NM','CAMPAIGN_ID','CAMPAIGN_NAME','POLYGON_NAME','SLOT']).agg({
        'SURGE_COEF':'mean', 'GEOMINIMAL_PRICE':'sum', 'DRIVER_BILL_AMT':'sum', 'DI_GEO_MINIMAL_AMT':'sum',
        'ORDER_RK':'count'}).reset_index()

    #считаем необходимые метрики:
    df['GP2DBNG'] = (df['GEOMINIMAL_PRICE'])/ ( df['DRIVER_BILL_AMT'] - df['DI_GEO_MINIMAL_AMT'])

    df['SLOT'] = pd.to_datetime(df['SLOT'])
    df['TIME'] = df['SLOT'].dt.strftime('%H:%M')

    df['OLD_PRICE'] = df['GEOMINIMAL_PRICE'] / df['ORDER_RK']

    df['TRIP_RANK'] = df.sort_values(['ORDER_RK'], ascending=False).groupby(['LOCALITY_NM']).cumcount() + 1
    df['TRIP_RANK_MAX'] = df.groupby(['LOCALITY_NM'])['TRIP_RANK'].transform('max')
    df['IS_TOP'] = df.apply(lambda x: 1 if x['TRIP_RANK'] <= round(x['TRIP_RANK_MAX']/3) else 0, axis = 1)

    #медиана по топ-33% наблюдений
    ltm = df[df['IS_TOP'] == 1].groupby(['LOCALITY_NM'])['GP2DBNG'].median().reset_index().rename(
        columns={'GP2DBNG': 'LOC_TOP_MEDIAN'})

    df = df.merge(ltm, on = 'LOCALITY_NM', how = 'left')

    #предложение новой цены
    df['REDUCTION_CALC'] = (df['GP2DBNG'] - df['LOC_TOP_MEDIAN']) / df['GP2DBNG']

    df['REDUCTION_SUGGEST'] = df.apply(lambda x: 0 if x['REDUCTION_CALC'] < 0 else (
        0.1 if x['REDUCTION_CALC'] > 0.1 else x['REDUCTION_CALC']), axis = 1)

    df['NEW_PRICE'] = round(df['OLD_PRICE']*(1 - df['REDUCTION_SUGGEST']), -1)

    #медиана трипов в городе
    df['LOC_TRIP_MEDIAN'] = df.groupby(['LOCALITY_NM'])['ORDER_RK'].transform('median')

    #Создаем словарь с названиями и id городов
    localities_info = {}
    for i in gh_data[['LOCALITY_NM','LOCALITY_RK']].drop_duplicates().values:
        localities_info.update({i[0]:i[1]})


    alert_df = df[(df['GP2DBNG'] > 2) & (df['SURGE_COEF'] < 1.5) &
                  (df['ORDER_RK'] > df['LOC_TRIP_MEDIAN'])]


    if len(alert_df) != 0:
        notifination.send_to_slack(":alert_slow: Обратите внимание на высокий GH :alert:")

        res = str()
        for l in alert_df.LOCALITY_NM.unique():
            l_df = alert_df[alert_df['LOCALITY_NM'] == l][['CAMPAIGN_ID', 'CAMPAIGN_NAME', 'POLYGON_NAME', 'TIME',
                                                           'GP2DBNG']]
            res = f':city: *{l}*\n'

            if l in r.managers.keys():
                responsibles = str()
                for i in r.managers.get(l).values():
                    responsibles += f'<@{i}> '
                res += f'{responsibles}\n'
            else:
                continue

            for c in l_df.CAMPAIGN_ID.unique():
                c_df = l_df[l_df['CAMPAIGN_ID'] == c]

                for p in c_df.POLYGON_NAME.unique():
                    p_df = c_df[c_df['POLYGON_NAME'] == p]

                    res += f'{p_df.CAMPAIGN_ID.values[0]}, {p_df.CAMPAIGN_NAME.values[0]}, {p_df.POLYGON_NAME.values[0]}:\n\t'

                    for row in p_df.values:
                        res += f'{row[3]} - GH {row[4]:.2f}, '

                    res += '\n'

            blocks = [
                {"type": "actions",
                 "elements": [
                     {
                         "type": "button",
                         "text": {
                             "type": "plain_text",
                             "text": "Taxiserv",
                             "emoji": True
                         },
                         "value": "click_me_123",
                         "url": f"https://city-mobil.ru/taxiserv/cityadmin/{localities_info.get(l)}/old/city/minimalki"
                     }
                 ]
                 }
            ]

        notifination.send_to_slack(res)
        notifination.send_to_slack("Странненькие геоминималочки", blocks)
