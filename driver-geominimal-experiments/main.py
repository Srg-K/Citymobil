import query as q
import config
import utils
from stat_tests import StatTests
from statsmodels.stats.power import tt_ind_solve_power
import pandas as pd
import time
import logging
import db


def main(cfg=config, params=config.PRIMARY_METRICS):
    start_time = time.time()
    schedule_df = db.get_mysql_conn_prod().fetch(q.q_1())
    stat_tests = StatTests(cfg)
    split_col = cfg.SPLIT_COL

    if (len(schedule_df) == 0):
        logging.info('No geominimal exps found')
        raise SystemExit

    campaigns = schedule_df['campaign_id'].unique().tolist()
    print(campaigns)

    for campaign_id in campaigns:
        print(campaign_id)
        if len(schedule_df) == 0:
            break
        timings, from_date, until_date, gch = utils.preprocess_gch(schedule_df, campaign_id)
        geomstat_df = db.get_clickhouse_conn().fetch(q.q_3_new_format(campaign_id))
        print('processed polygons & geominimals')

        if (len(geomstat_df) == 0):
            logging.info(f"No data at tables: geominimal_stat; "
                         f"(campaign_id = {campaign_id})")
            continue

        driver_splits_df = db.get_mysql_conn_prod().fetch(q.q_4(campaign_id, gch['id_locality'][0]))
        print('driver')
        # получение саджестов, которые были в геоминимальных полигонах
        suggest_df = utils.get_raw_data(
            campaign_id=campaign_id,
            locality=gch['id_locality'][0],
            timings=timings,
            from_date=from_date,
            until_date=until_date,
            gch=gch,
            geomstat_df=geomstat_df,
            driver_splits_df=driver_splits_df
            )
        if suggest_df[(suggest_df['retro'] == 'tw')].split.nunique() < 2:
            logging.info(f'Cant find splits. Closing the process')
            continue
        suggest_df, polygon_df = utils.preprocess(suggest_df, gch)

        suggest_df['rn'] = suggest_df.groupby('fss_id').cumcount() + 1

        exp_res = []
        totals_res = []
        stats_res = []
        combs = utils.combine_splits(suggest_df, split_col)
        for comb in combs:
            print(comb)
            df_with_single_combination = suggest_df[suggest_df[split_col].isin(comb)]
            stats, totals = utils.get_totals(df=df_with_single_combination, campaign_id=campaign_id, split_col=split_col,
                                             split_pair=comb)
            stats_res.append(stats)
            totals_res.append(totals)

            for metric in params:
                print(metric)
                df_compressed = utils.compress_and_filter(
                    df=df_with_single_combination,
                    to_compress=params[metric]['to_compress'],
                    estimator=params[metric]["estimator"],
                    options=params[metric]['options'],
                    filter_to=params[metric]['filter_to'],
                    split_col=params[metric]['split_col']
                )
                exp_res.append(
                    stat_tests.stat_inference(
                        df=df_compressed,
                        split_col=params[metric]["split_col"],
                        metric_col=params[metric]["metric_col"],
                        estimator=params[metric]["estimator"],
                        metric=metric,
                        split_pair=comb,
                        post_power=params[metric]["post_power"]
                    )
                )
        exp_res = stat_tests.multiple_comparison(pd.concat(exp_res))
        totals_res = pd.concat(totals_res)
        df_to_db = pd.concat(stats_res).merge(exp_res, on=['metric', 'split_pair'], how="left")

        df_to_db['mde'] = tt_ind_solve_power(
            alpha=1 - config.CONFIDENCE_LEVEL, power=config.POWER_LEVEL, nobs1=totals_res['drivers'][0],
            ratio=totals_res['drivers'][0] / totals_res['drivers'][1])

        utils.upload_to_db({'stats': df_to_db, 'totals': pd.concat([totals_res])}, campaign_id)
        logging.info(f'uploaded {campaign_id} with time {time.time() - start_time}')

main()
