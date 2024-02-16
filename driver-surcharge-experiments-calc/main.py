import pandas as pd
import config
import utils
import db
import query as q
import logging
import time
from stat_tests import StatTests


def main(cfg=config, params=config.PRIMARY_METRICS):

    stat_tests = StatTests(cfg)
    surcharge_info = db.get_mysql_conn_prod().fetch(q.q_1())
    print(surcharge_info)

    for _, exp in surcharge_info.iterrows():
        print(exp['id'])
        try:
            df = utils.get_raw_data(
                sur_id=exp['id'],
                date_start=exp['date_start'],
                date_end=exp['date_end']
            )
            if df is None:
                continue
        except Exception as e:
            continue

        exp_res = []
        totals_res = []
        for comb in utils.combine_splits(df, 'split'):
            df_with_single_combination = df[df.split.isin(comb)]
            totals = utils.get_totals(df=df_with_single_combination, split_col='split', sur_id=exp['id'], split_pair=comb)
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
        df_to_db = pd.concat(totals_res).merge(exp_res, on=['metric', 'split_pair'], how="left")
        utils.upload_to_db(df=df_to_db, sur_id=exp["id"])
    return


if __name__ == '__main__':
    start_time = time.time()
    main()
    logging.info(f'uploaded with time {time.time() - start_time}')
