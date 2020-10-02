# -*- coding: utf-8 -*-
"""
 @File    : demo79.py
 @Time    : 2020/7/28 下午10:29
 @Author  : yizuotian
 @Description    :
"""
import os

import catboost as cb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split


def load_data(data_root):
    movie_path = os.path.join(data_root, 'itemInfo.csv')
    user_path = os.path.join(data_root, 'userInfo.csv')
    train_path = os.path.join(data_root, 'train.csv')
    test_path = os.path.join(data_root, 'test.csv')

    movies = pd.read_csv(movie_path, parse_dates=['release_date'])
    movies['year'] = movies['release_date'].dt.year.astype('str')
    movies['month'] = movies['release_date'].dt.month.astype('str')
    movies['week'] = movies['release_date'].dt.week.astype('str')
    movies = movies.drop(columns=['Unnamed: 0', 'unknowngenres'])

    users = pd.read_csv(user_path)
    users['z1'] = users.useZipcode.str[0:1]
    # users['z2'] = users.useZipcode.str[0:3]
    # users['z3'] = users.useZipcode.str[0:5]

    # 训练数据加载
    train = pd.read_csv(train_path, header=None, names=['userId', 'movie_id', 'time', 'score'])
    train.time = pd.to_datetime(train.time, unit='s')
    train['w_year'] = train.time.dt.year.astype('str')
    train['w_month'] = train.time.dt.month.astype('str')
    train['w_week'] = train.time.dt.week.astype('str')

    # train = train.apply(get_watch_time, axis=1)

    test = pd.read_csv(test_path, header=None, names=['userId', 'movie_id', 'time'])
    origin_test = test.copy()
    test.time = pd.to_datetime(test.time, unit='s')
    test['w_year'] = test.time.dt.year.astype('str')
    test['w_month'] = test.time.dt.month.astype('str')
    test['w_week'] = test.time.dt.week.astype('str')

    # 统计用户评分最大最小值和均值
    user_stat = train.groupby('userId')['score'].agg(['max', 'min', 'mean', 'std'])
    movie_stat = train.groupby('movie_id')['score'].agg(['max', 'min', 'mean', 'std'])

    train = pd.merge(train, user_stat, how='left', left_on='userId',
                     right_index=True, suffixes=('', '_user'))
    train = pd.merge(train, movie_stat, how='left', left_on='movie_id',
                     right_index=True, suffixes=('', '_movie'))

    test = pd.merge(test, user_stat, how='left', left_on='userId',
                    right_index=True, suffixes=('', '_user'))
    test = pd.merge(test, movie_stat, how='left', left_on='movie_id',
                    right_index=True, suffixes=('', '_movie'))

    train['us_delta'] = (train['score'] - train['mean']) / train['std']
    train['ms_delta'] = (train['score'] - train['mean_movie']) / train['std_movie']
    train['ms_delta'] = train['ms_delta'].fillna(0)

    # test = test.rename(columns={'score': 'score_max'})

    train = pd.merge(train, movies, left_on='movie_id', right_on='movie_id', sort=True)
    train = pd.merge(train, users, left_on='userId', right_on='userId', sort=True)

    test = pd.merge(test, movies, left_on='movie_id', right_on='movie_id', sort=True)
    test = pd.merge(test, users, left_on='userId', right_on='userId', sort=True)

    train['w_days'] = (train.time - train.release_date).dt.days
    test['w_days'] = (test.time - test.release_date).dt.days

    """
    train_accuracy：0.430 
    test_accuracy：0.407
    """
    # y = train['ms_delta']
    # y = train['us_delta']
    y = train['score']

    assert np.alltrue(test['userId'] == origin_test['userId'])
    assert np.alltrue(test['movie_id'] == origin_test['movie_id'])

    x = train.drop(columns=['score', 'time', 'us_delta', 'ms_delta',  # train
                            'useZipcode',  # user
                            'userId', 'movie_id',
                            # 'movie_title',
                            'release_date', 'video_release_date', 'IMDb_URL'])

    test = test.drop(columns=['time',  # test
                              'useZipcode',  # user
                              'userId', 'movie_id',
                              # 'movie_title',
                              'release_date', 'video_release_date', 'IMDb_URL'])

    # test.columns = x.columns
    print(x.columns)
    print(test.columns)

    return x, y, test, origin_test


def cls_boost(data_root, out_csv_file):
    """
    train_accuracy： 0.5401527777777778
    test_accuracy： 0.471

    :param data_root:
    :param out_csv_file:
    :return:
    """

    x, y, test, origin_test = load_data(data_root)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)

    cat_index = [i for i, col in enumerate(x.columns) if col in
                 ['useGender', 'useOccupation', 'z1', 'z2', 'z3',
                  'userId', 'movie_id',
                  'year', 'month', 'week',
                  'w_year', 'w_month', 'w_week']]

    text_index = [i for i, col in enumerate(x.columns) if col in
                 ['movie_title']]

    model = cb.CatBoostClassifier(iterations=1000, learning_rate=0.1,
                                  od_type="Iter", l2_leaf_reg=3,
                                  text_features=text_index,
                                  depth=10, cat_features=cat_index)

    model.fit(x_train, y_train, eval_set=(x_test, y_test))

    k = pd.DataFrame(metrics.confusion_matrix(y_test, model.predict(x_test)))
    print('confusion_matrix:\n {}'.format(k))
    print("train_accuracy：", model.score(x_train, y_train), "\n",
          "test_accuracy：", model.score(x_test, y_test))

    save_feature_importance(model, os.path.join(data_root, 'feature_cls.png'))

    origin_test['score'] = model.predict(test)
    origin_test[['userId', 'movie_id', 'time', 'score']].to_csv(os.path.join(data_root, out_csv_file),
                                                                sep=',', index=False, header=None)


def rgr_boost(data_root, out_csv_file, delta=True):
    """
    # 去除邮编
    train_accuracy：0.457
     test_accuracy：0.435
      a)保留z1
    train_accuracy：0.454
     test_accuracy：0.436

    # 增加标准差
    train_accuracy：0.457
     test_accuracy：0.436

    # 去除userId,movie_id
    train_accuracy：0.455
     test_accuracy：0.431


    :param data_root:
    :param out_csv_file:
    :param delta:
    :return:
    """
    x, y, test, origin_test = load_data(data_root)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)

    cat_index = [i for i, col in enumerate(x.columns) if col in
                 ['useGender', 'useOccupation', 'z1', 'z2', 'z3',
                  'userId', 'movie_id',
                  'year', 'month', 'week',
                  'w_year', 'w_month', 'w_week']]
    text_indx = [i for i, col in enumerate(x.columns) if col in
                 ['movie_title']]

    model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1,
                                 od_type="Iter", l2_leaf_reg=3, model_size_reg=3,
                                 depth=10,

                                 cat_features=cat_index)

    model.fit(x_train, y_train, eval_set=(x_test, y_test))

    print("train_accuracy：{:.3f} \n"
          " test_accuracy：{:.3f}".format(accuracy(model, x_train, y_train, delta),
                                         accuracy(model, x_test, y_test, delta)))
    save_feature_importance(model, os.path.join(data_root, 'feature_rgr.png'))

    origin_test['score'] = predict(model, test, delta).round().astype('int32')
    origin_test[['userId', 'movie_id', 'time', 'score']].to_csv(os.path.join(data_root, out_csv_file),
                                                                sep=',', index=False, header=None)


def accuracy(model, x, y, delta=True):
    y_predict = predict(model, x, delta)
    y = y * x['std'] + x['mean']
    return np.mean(y.round().astype('int32') == y_predict.round().astype('int32'))


def predict(model, x, delta=True):
    y_predict = model.predict(x)
    if delta:
        y_predict = y_predict * x['std'] + x['mean']

    return y_predict


def save_feature_importance(model, png_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.barh(model.feature_names_, model.feature_importances_, height=1.)
    plt.savefig(png_path, format='png')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output-dir', default='./', help='path where to save')
    # arguments = parser.parse_args(sys.argv[1:])
    #
    # main(arguments)
    # rgr_boost('AIyanxishe_79', 'demo79.rgr.csv')
    cls_boost('AIyanxishe_79', 'demo79.cls.csv')
