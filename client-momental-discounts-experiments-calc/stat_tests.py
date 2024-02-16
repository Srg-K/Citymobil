import pandas as pd
import numpy as np
import config
import random
import logging
from scipy import stats
from scipy.stats import norm, rankdata
from statsmodels.stats.multitest import multipletests


class StatTests(object):
    """Класс статистических методов
    """

    def __init__(
            self,
            cfg: config
    ):
        """

        Parameters
        ----------
        cfg
            simulation_num - количество симуляций
            percent_sample - доля наблюдений для подвыборки из исходной выборки
            max_n_sample – ограничение по количеству наблюдений в подвыборке
            confidence_level – уровень значимости
            boot_it – количество бутстрап итераций
        """

        self.simulation_num: int = cfg.SIMULATION_NUM
        self.percent_sample: float = cfg.PERCENT_SAMPLE
        self.max_n_sample: int = cfg.MAX_N_SAMPLE
        self.confidence_level: float = cfg.CONFIDENCE_LEVEL
        self.boot_it: int = cfg.BOOT_ITERATIONS

    def fpr(self, df_x: pd.DataFrame, df_y: pd.DataFrame, metric_col: str, estimator: str, *args, **kwargs):

        """
        Расчет FPR для A/A-теста (или TPR для A/B)

        Parameters
        ----------
        df_x : pd.DataFrame
            Таблица данных эксперимента первого сплита X
        df_y : pd.DataFrame
            Таблица данных эксперимента второго сплита Y
        metric_col : str
            Наименование показателя для проверки
        estimator : {'ttest', 'prop', 'bootstrap'}
            Статистический критерий
        *args
        **kwargs

        Returns
        -------
        out : dict
            Результаты многократного расчета pvalue,
            средних и итоговый показатель FPR/TPR.
        """
        stat_res = {
            'aa': {'pvalue': []},
            'fpr': {f'fpr_{self.confidence_level}': .0}
        }

        estimators = {
            'ttest': self.t_test,
            'ttest_ratio': self.ttest_ratio,
            'prop': self.prop_test,
            'bootstrap': self.bootstrap
        }

        for sim in range(self.simulation_num):
            if estimator == 'ttest_ratio':  # для остальных критериев надо написать свои правила
                metric_col = None
            x = self.make_sample(df_x, metric_col)
            y = self.make_sample(df_y, metric_col)
            pvalue = estimators[estimator](x, y, *args, **kwargs)
            stat_res['aa']['pvalue'].append(pvalue)
        stat_res['fpr'][f'fpr_{self.confidence_level}'] = self.compute_fpr(stat_res['aa']['pvalue'])

        return stat_res

    def t_test(self, x, y, *args, **kwargs):
        try:
            _, pvalue = stats.ttest_ind(x[~np.isnan(x)], y[~np.isnan(y)], *args, **kwargs)
        except ValueError as e:
            print('pvalue error',e)
            return None
        return pvalue

    def ttest_ratio(self, df_0, df_1):
        x_0 = np.array(df_0['x'])
        x_1 = np.array(df_1['x'])
        y_0 = np.array(df_0['y'])
        y_1 = np.array(df_1['y'])
        n_0 = np.array(df_0['n'].sum())
        n_1 = np.array(df_1['n'].sum())

        if y_0.sum() == 0 or y_1.sum() == 0:
            pvalue = [np.float64(1)]
        else:
            mean_x_0 = x_0.sum() / n_0
            mean_x_1 = x_1.sum() / n_1
            mean_y_0 = y_0.sum() / n_0
            mean_y_1 = y_1.sum() / n_1
            var_x_0 = np.sum(np.array([abs(a - mean_x_0)**2 for a in x_0])) / n_0
            var_x_1 = np.sum(np.array([abs(a - mean_x_1)**2 for a in x_1])) / n_1
            var_y_0 = np.sum(np.array([abs(a - mean_y_0)**2 for a in y_0])) / n_0
            var_y_1 = np.sum(np.array([abs(a - mean_y_1)**2 for a in y_1])) / n_1
            cov_0 = np.sum((x_0 - mean_x_0.reshape(-1, 1)) * (y_0 - mean_y_0.reshape(-1, 1)), axis=1) / n_0
            cov_1 = np.sum((x_1 - mean_x_1.reshape(-1, 1)) * (y_1 - mean_y_1.reshape(-1, 1)), axis=1) / n_1

            var_0 = var_x_0 / mean_y_0 ** 2 + var_y_0 * mean_x_0 ** 2 / mean_y_0 ** 4 - 2 * mean_x_0 / mean_y_0 ** 3 * cov_0
            var_1 = var_x_1 / mean_y_1 ** 2 + var_y_1 * mean_x_1 ** 2 / mean_y_1 ** 4 - 2 * mean_x_1 / mean_y_1 ** 3 * cov_1

            rto_0 = np.sum(x_0) / np.sum(y_0)
            rto_1 = np.sum(x_1) / np.sum(y_1)
            statistic = (rto_1 - rto_0) / np.sqrt(var_0 / n_0 + var_1 / n_1)
            pvalue = 2 * np.minimum(stats.norm(0, 1).cdf(statistic), 1 - stats.norm(0, 1).cdf(statistic))
        return pvalue[0]

    def prop_test(self, x, y, *args, **kwargs):
        counts = np.array([sum(x), sum(y)])
        nobs = np.array([len(x), len(y)])
        try:
            _, pvalue = stats.proportions_ztest(counts, nobs, *args, **kwargs)
        except ValueError as e:
            logging.error(e)
            return None
        return pvalue

    def bootstrap(self, x, y):
        boot_dist = []
        for sim in range(self.simulation_num):
            xb = np.mean(random.choice(x))
            yb = np.mean(random.choice(y))
            boot_dist.append(yb - xb)
        try:
            pvalue = min(
                norm.cdf(0, np.mean(boot_dist), np.std(boot_dist)),
                norm.cdf(0, -np.mean(boot_dist), np.std(boot_dist))
            ) * 2
        except ValueError as e:
            logging.error(e)
            return None
        return pvalue

    def make_sample(self, df: pd.DataFrame, metric_col: str = None):
        """
        Генерация подвыборки

        Parameters
        ----------
        df
        metric_col

        Returns
        -------

        """
        sample_size = int(len(df) * self.percent_sample)
        if metric_col is not None:
            df = df[metric_col].sample(sample_size, replace=False).values
        else:
            df = df.sample(sample_size, replace=False)
        return df

    def compute_fpr(self, pvalues: list):
        """
        Доля pvalues, соответствующих условию проверки

        Parameters
        ----------
        pvalues

        Returns
        -------

        """
        try:
            return float(sum(np.array(pvalues) <= 1 - self.confidence_level) / self.simulation_num)
        except ValueError as e:
            logging.error(e)
            return None

    def fpr_by_group(
            self,
            df: pd.DataFrame,
            split_col: str,
            group_col: str,
            metric_col: str,
            *args,
            **kwargs
    ):
        """
        Запуск расчета fpr/tpr по группам.

        Parameters
        ----------
        df : pd.DataFrame
            Таблица с сырыми данными эксперимента
        metric_col : str
            Наименование показателя для проверки
        split_col : str
            Наименование колонки признака сплита
        group_col : str
            Наименование колонки для группировки внутри сплитов
        *args
        **kwargs

        Returns
        -------
        out : pd.DataFrame
            Таблица с результатами расчета FPR/TPR
            Если в параметрах не было необходимых групп, то df.group = None
        """

        report_data = []
        split_x, split_y = [x for _, x in df.groupby(split_col)]

        if len(split_x) < 20 or len(split_y) < 20:
            pass

        if group_col is None or len(group_col) == 0:
            fpr_value = self.fpr(split_x, split_y, metric_col, *args, **kwargs)
            report_data.append(
                [
                    None,
                    split_x[metric_col].mean(),
                    split_y[metric_col].mean(),
                    fpr_value['fpr'][f'fpr_{self.confidence_level}'],
                ]
            )

        else:
            group_names = list(pd.unique(df[group_col]))
            for name in group_names:
                df_x = split_x[(split_x[group_col] == name)]
                df_y = split_y[(split_y[group_col] == name)]
                fpr_value = self.fpr(df_x, df_y, metric_col, *args, **kwargs)
                report_data.append(
                    [
                        name,
                        df_x[metric_col].mean(),
                        df_y[metric_col].mean(),
                        fpr_value['fpr'][f'fpr_{self.confidence_level}'],
                    ]
                )

        report = pd.DataFrame.from_records(report_data,
                                           columns=['group', 'mu_x', 'mu_y', f'fpr_{self.confidence_level}'])
        return report

    def stat_inference(
            self,
            df: pd.DataFrame,
            split_col: str,
            metric_col: str,
            estimator: str,
            metric: str,
            split_pair: list,
            post_power: bool = True

    ):
        """
        Расчет значимости для метрики

        Parameters
        ----------
        df : pd.DataFrame
        split_col : str
            Наименование колонки со сплитами
        metric_col : str
            Наименование числовой колонки с метрикой для проверки
        estimator : str
            Наименование стат. критерия (ttest, prop, bootstrap)
        metric : str
            Наименование метрики в конфиге (tpd, average_bil, etc.)
        post_power : bool
            Флаг расчета мощности

        Returns
        -------
        out : pd.DataFrame
            Результат расчета значимости
            - pvalue
            - is_significant

        """
        stat_res = {
            "split_pair": '_'.join(str(i) for i in split_pair),
            "metric": metric,
            "pvalue": float,
            "power": None,
            "is_significant": bool
        }

        estimators = {
            'ttest': self.t_test,
            'ttest_ratio': self.ttest_ratio,
            'prop': self.prop_test,
            'bootstrap': self.bootstrap
        }

        split_x, split_y = [x for _, x in df.groupby(split_col)]
        if len(split_x) < 20 or len(split_y) < 20:
            pass

        if estimator == 'ttest_ratio':
            stat_res['pvalue'] = estimators[estimator](split_x, split_y)
        else:
            stat_res['pvalue'] = estimators[estimator](split_x[metric_col], split_y[metric_col])

        if post_power == 1 or post_power == True:
            res = self.fpr(split_x, split_y, metric_col, estimator)
            stat_res['power'] = res['fpr'][f'fpr_{config.CONFIDENCE_LEVEL}']
        else:
            stat_res['power'] = 0.0

        stat_res['is_significant'] = stat_res['pvalue'] < 1 - config.CONFIDENCE_LEVEL
        stat_res['is_significant'] = stat_res['is_significant'].astype('uint8')

        # pd.DataFrame({'pvalue': res['aa']['pvalue']}).to_csv(f'csvs/pvalue_res_{metric}.csv')
        return pd.DataFrame([stat_res])

    def multiple_comparison(self,df: pd.DataFrame):
        """
        Коррекция pvalue внутри эксперимента

        Parameters
        ----------
        df

        Returns
        -------
        out : pd.DataFrame
            Результаты коррекции
            - pvalue_adj
            - is_significant_adj
        """
        df['pvalue_adj'], df['is_significant_adj'] = pd.Series(dtype=float), pd.Series(dtype=int)
        df.loc[df['pvalue'].notna(), ['pvalue_adj']] = \
            multipletests(df['pvalue'][df['pvalue'].isna() == False],
                          alpha=1 - config.CONFIDENCE_LEVEL, method='fdr_bh')[1]
        df['is_significant_adj'] = df['pvalue_adj'] < 1 - config.CONFIDENCE_LEVEL
        df['is_significant_adj'] = df['is_significant_adj'].astype(int)
        return df

