# -*- coding: utf-8 -*-
"""
 @File    : railway.py
 @Time    : 2020/10/19 上午10:21
 @Author  : yizuotian
 @Description    :
"""
import datetime
import os

import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_root):
    train_csv = os.path.join(data_root, 'train.csv')
    test_csv = os.path.join(data_root, 'test.csv')
    train_df = pd.read_csv(train_csv, parse_dates=['datetime'])
    test_df = pd.read_csv(test_csv, parse_dates=['datetime'])

    train_df['year'] = train_df.datetime.dt.year
    train_df['mon'] = train_df.datetime.dt.month
    train_df['day'] = train_df.datetime.dt.day
    train_df['hour'] = train_df.datetime.dt.hour
    train_df['week'] = train_df.datetime.dt.week
    train_df['weekday'] = train_df.datetime.dt.weekday

    test_df['year'] = test_df.datetime.dt.year
    test_df['mon'] = test_df.datetime.dt.month
    test_df['day'] = test_df.datetime.dt.day
    test_df['hour'] = test_df.datetime.dt.hour
    test_df['week'] = test_df.datetime.dt.week
    test_df['weekday'] = test_df.datetime.dt.weekday

    train_df_am = train_df.query('hour < 12')
    train_df_pm = train_df.query('hour >= 12')

    test_df_am = test_df.query('hour < 12')
    test_df_pm = test_df.query('hour >= 12')

    am_stat = train_df_am.groupby(['year', 'mon', 'day'])['cnt'].agg(['max', 'min', 'mean', 'std'])
    pm_stat = train_df_pm.groupby(['year', 'mon', 'day'])['cnt'].agg(['max', 'min', 'mean', 'std'])

    # 训练样本
    train_df_pm = train_df_pm[train_df_pm.datetime < datetime.datetime.strptime('2014-06-25', '%Y-%m-%d')]
    train_df_am = train_df_am[train_df_am.datetime < datetime.datetime.strptime('2014-06-25', '%Y-%m-%d')]
    am_to_pm = pd.merge(train_df_pm, am_stat, how='left', left_on=['year', 'mon', 'day'], right_index=True)
    pm_to_am = pd.merge(train_df_am, pm_stat, how='left', left_on=['year', 'mon', 'day'], right_index=True)

    train = pd.concat([am_to_pm, pm_to_am])

    # 测试样本
    test_am_to_pm = pd.merge(test_df_pm, am_stat, how='left', left_on=['year', 'mon', 'day'], right_index=True)
    test_pd_to_am = pd.merge(test_df_am, pm_stat, how='left', left_on=['year', 'mon', 'day'], right_index=True)
    test = pd.concat([test_am_to_pm, test_pd_to_am])

    return train, test


def main(data_root, out_csv_file):
    train, origin_test = load_data(data_root)
    y = train['cnt']
    x = train.drop(columns=['cnt'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)

    cat_index = [i for i, col in enumerate(x.columns) if col in
                 ['hour', 'month', 'day', 'week', 'weekday']]

    model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1,
                                 od_type="Iter", l2_leaf_reg=3, model_size_reg=None,
                                 depth=6,
                                 cat_features=cat_index)

    model.fit(x_train, y_train, eval_set=(x_test, y_test))

    test = origin_test.drop(columns=['id'])
    origin_test['cnt'] = model.predict(test).round().astype('int')

    origin_test = origin_test.sort_values('id')

    origin_test[['id', 'cnt']].to_csv(os.path.join(data_root, out_csv_file),
                                      sep=',', index=False, header=None)


if __name__ == '__main__':
    main('high_speed_rail', 'rst_railway.csv')
    # import datetime.date
