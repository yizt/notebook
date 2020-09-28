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
import torch
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
    users['z2'] = users.useZipcode.str[0:3]
    users['z3'] = users.useZipcode.str[0:5]

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

    train = pd.merge(train, movies, left_on='movie_id', right_on='movie_id', sort=True)
    train = pd.merge(train, users, left_on='userId', right_on='userId', sort=True)

    test = pd.merge(test, movies, left_on='movie_id', right_on='movie_id', sort=True)
    test = pd.merge(test, users, left_on='userId', right_on='userId', sort=True)

    train['w_days'] = (train.time - train.release_date).dt.days
    test['w_days'] = (test.time - test.release_date).dt.days

    y = train['score']

    assert np.alltrue(test['userId'] == origin_test['userId'])
    assert np.alltrue(test['movie_id'] == origin_test['movie_id'])

    x = train.drop(columns=['score', 'time',  # train
                            'useZipcode',  # user
                            'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'])

    test = test.drop(columns=['time',  # test
                              'useZipcode',  # user
                              'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'])

    return x, y, test, origin_test


def cls_boost(data_root, out_csv_file):
    """
    train_accuracy： 0.4845277777777778
     test_accuracy： 0.441
    :param data_root:
    :param out_csv_file:
    :return:
    """

    x, y, test, origin_test = load_data(data_root)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)

    cat_index = [i for i, col in enumerate(x.columns) if col in
                 ['useGender', 'useOccupation', 'z1', 'z2', 'z3',
                  'year', 'month', 'week',
                  'w_year', 'w_month', 'w_week']]

    model = cb.CatBoostClassifier(iterations=1000, learning_rate=0.1,
                                  od_type="Iter", l2_leaf_reg=3,
                                  depth=6, cat_features=cat_index)

    model.fit(x_train, y_train, eval_set=(x_test, y_test))

    k = pd.DataFrame(metrics.confusion_matrix(y_test, model.predict(x_test)))
    print('confusion_matrix:{}'.format(k))
    print("train_accuracy：", model.score(x_train, y_train), "\n",
          "test_accuracy：", model.score(x_test, y_test))

    save_feature_importance(model, os.path.join(data_root, 'feature_cls.png'))

    origin_test['score'] = model.predict(test)
    origin_test[['userId', 'movie_id', 'time', 'score']].to_csv(os.path.join(data_root, out_csv_file),
                                                                sep=',', index=False, header=None)


def rgr_boost(data_root, out_csv_file):
    """
    train_accuracy：0.459
     test_accuracy：0.416
    :param data_root:
    :param out_csv_file:
    :return:
    """
    x, y, test, origin_test = load_data(data_root)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)

    cat_index = [i for i, col in enumerate(x.columns) if col in
                 ['useGender', 'useOccupation', 'z1', 'z2', 'z3',
                  'year', 'month', 'week',
                  'w_year', 'w_month', 'w_week']]

    model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1,
                                 od_type="Iter", l2_leaf_reg=3, model_size_reg=3,
                                 depth=10,
                                 cat_features=cat_index)

    model.fit(x_train, y_train, eval_set=(x_test, y_test))

    print("train_accuracy：{:.3f} \n"
          " test_accuracy：{:.3f}".format(np.mean(y_train == model.predict(x_train).round().astype('int32')),
                                         np.mean(y_test == model.predict(x_test).round().astype('int32'))))
    save_feature_importance(model, os.path.join(data_root, 'feature_rgr.png'))

    origin_test['score'] = model.predict(test).round().astype('int32')
    origin_test[['userId', 'movie_id', 'time', 'score']].to_csv(os.path.join(data_root, out_csv_file),
                                                                sep=',', index=False, header=None)


def save_feature_importance(model, png_path):
    import matplotlib.pyplot as plt
    plt.barh(model.feature_names_, model.feature_importances_)
    plt.savefig(png_path, format='png')


def accuracy(net, data_loader):
    net.eval()
    acc_list = []
    for users, movies, scores, _ in data_loader:
        scores_pred = net(users, movies)
        scores_pred *= data_loader.dataset.std
        scores_pred += data_loader.dataset.mean
        acc = (torch.round(scores_pred) == scores).numpy()
        acc_list.append(acc)

    acc_list = np.concatenate(acc_list, axis=0)
    return np.mean(acc_list)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output-dir', default='./', help='path where to save')
    # arguments = parser.parse_args(sys.argv[1:])
    #
    # main(arguments)
    # rgr_boost('AIyanxishe_79', 'demo79.rgr.csv')
    cls_boost('AIyanxishe_79', 'demo79.cls.csv')
