# -*- coding: utf-8 -*-
"""
 @File    : demo79.py
 @Time    : 2020/7/28 下午10:29
 @Author  : yizuotian
 @Description    :
"""
import codecs
import os
import time

import catboost as cb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def replace_dot_in_double_quote(string):
    """
    将双引号里面的逗号替换为中文的
    :param string:
    :return:
    """
    char_list = []
    in_double_quote = False
    for c in string:
        if c == ',' and in_double_quote:
            c = '，'
        if c == '\"':
            in_double_quote = not in_double_quote
        char_list.append(c)
    return ''.join(char_list)


def load_movies(csv_path):
    with codecs.open(csv_path) as f:
        lines = f.readlines()
    movie_features = []
    for line in lines[1:]:
        # print(len(line.strip().split(',')))
        line = replace_dot_in_double_quote(line.strip())
        _, movie_id, movie_title, release_date, video_release_date, imdb_url, unknowngenres, action, adventure, \
        animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror, musical, mystery, \
        romance, sci_fi, thriller, war, western = line.strip().split(',')
        movie_features.append(
            [unknowngenres, action, adventure, animation, childrens, comedy, crime, documentary, drama,
             fantasy,
             film_noir, horror, musical, mystery, romance, sci_fi, thriller, war, western])

    return np.array(movie_features).astype(np.float32)


def load_users(csv_path):
    with codecs.open(csv_path) as f:
        lines = f.readlines()

    user_features = []
    for line in lines[1:]:
        userId, useAge, useGender, useOccupation, useZipcode = line.split(',')
        user_features.append([useAge, useGender, useOccupation, useZipcode[0]])

    user_features = np.array(user_features)
    age = user_features[:, 0].astype(np.float32)
    mean = np.mean(age)
    std = np.std(age)
    age -= mean
    age /= std

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(user_features[:, 1:])
    f2 = enc.transform(user_features[:, 1:]).toarray()

    return np.concatenate([age[:, np.newaxis], f2], axis=1).astype(np.float32)


def load_train(csv_path):
    with codecs.open(csv_path) as f:
        lines = f.readlines()
    train_list = []
    for line in lines:
        user_id, movie_id, _, score = line.strip().split(',')
        train_list.append([user_id, movie_id, score])
    train_samples = np.array(train_list).astype(np.int)
    return train_samples


def load_test(csv_path):
    with codecs.open(csv_path) as f:
        lines = f.readlines()
    train_list = []
    for line in lines:
        user_id, movie_id, _ = line.strip().split(',')
        train_list.append([user_id, movie_id])
    return np.array(train_list).astype(np.int)


def get_date_detail(movie):
    release_date = movie['release_date']
    movie['year'] = release_date.year
    movie['month'] = release_date.month
    movie['week'] = release_date.week
    movie['day'] = release_date.day
    return movie


def get_watch_time(rec):
    watch_time = rec['time']
    watch_time = time.localtime(watch_time)
    rec['year'] = watch_time.tm_year
    rec['month'] = watch_time.tm_mon
    rec['mday'] = watch_time.tm_mday
    rec['wday'] = watch_time.tm_wday
    return rec


def load_data(data_root):
    movie_path = os.path.join(data_root, 'itemInfo.csv')
    user_path = os.path.join(data_root, 'userInfo.csv')
    train_path = os.path.join(data_root, 'train.csv')
    test_path = os.path.join(data_root, 'test.csv')

    movies = pd.read_csv(movie_path, parse_dates=['release_date'])
    movies['year'] = movies['release_date'].dt.year.astype('str')
    movies['month'] = movies['release_date'].dt.month.astype('str')
    movies['week'] = movies['release_date'].dt.week.astype('str')
    movies = movies.drop(columns=['Unnamed: 0'])
    # movies = movies.apply(get_date_detail, axis=1)

    users = pd.read_csv(user_path)
    users['z1'] = users.useZipcode.str[0:1]
    users['z2'] = users.useZipcode.str[1:3]
    users['z3'] = users.useZipcode.str[3:]

    train = pd.read_csv(train_path, header=None, names=['userId', 'movie_id', 'time', 'score'])
    train.time = pd.to_datetime(train.time, unit='s')
    train['w_year'] = train.time.dt.year.astype('str')
    train['w_year'] = train.time.dt.month.astype('str')
    train['w_week'] = train.time.dt.week.astype('str')
    # train = train.apply(get_watch_time, axis=1)

    test = pd.read_csv(test_path, header=None, names=['userId', 'movie_id', 'time'])
    origin_test = test.copy()
    test.time = pd.to_datetime(test.time, unit='s')
    test['w_year'] = test.time.dt.year
    test['w_year'] = test.time.dt.month
    test['w_week'] = test.time.dt.week

    train = pd.merge(train, users, left_on='userId', right_on='userId')
    train = pd.merge(train, movies, left_on='movie_id', right_on='movie_id')

    test = pd.merge(test, users, left_on='userId', right_on='userId')
    test = pd.merge(test, movies, left_on='movie_id', right_on='movie_id')

    y = train['score']
    x = train.drop(columns=['score', 'time',  # train
                            'useZipcode',  # user
                            'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'])

    test = test.drop(columns=['time',  # test
                              'useZipcode',  # user
                              'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'])

    return x, y, test, origin_test


def cat_boost_train():
    x, y, test, origin_test = load_data('AIyanxishe_79')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)

    cat_index = [i for i, col in enumerate(x.columns) if col in
                 ['useGender', 'useOccupation', 'z1', 'z2', 'z3',
                  'year', 'month', 'week',
                  'w_year', 'w_month', 'w_week']]

    model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, max_depth=6,
                                 model_size_reg=3, l2_leaf_reg=3,
                                 custom_metric='MAE')
    model.fit(x_train, y_train, eval_set=(x_test, y_test), cat_features=cat_index)

    origin_test['score'] = model.predict(test)

    origin_test[['userId', 'movie_id', 'time', 'score']].to_csv('./submit_79.csv', sep=',', index=False, header=None)


def random_result(csv_file):
    with codecs.open(csv_file) as f:
        lines = f.readlines()

    with codecs.open('./r79.csv', mode='w', encoding='utf-8') as w:
        for line in lines:
            line = "{},{}".format(line.strip(), np.random.randint(1, 6))
            w.write('{}\n'.format(line))


class YxsDataset(Dataset):
    def __init__(self, data_root='/Users/admin/Downloads/AIyanxishe_79', mode='train'):
        self.movies = load_movies(os.path.join(data_root, 'itemInfo.csv'))
        self.users = load_users(os.path.join(data_root, 'userInfo.csv'))
        self.train_samples = load_train(os.path.join(data_root, 'train.csv'))
        self.scores = self.train_samples[:, -1].astype(np.float32)
        self.mean = np.mean(self.scores)
        self.std = np.std(self.scores)
        self.scores -= self.mean
        self.scores /= self.std
        self.test_samples = load_test(os.path.join(data_root, 'test.csv'))
        self.mode = mode

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        if self.mode == 'train':
            user_id, movie_id, score = self.train_samples[item]

            user_features = self.users[user_id - 1]
            move_features = self.movies[movie_id - 1]
            score_norm = self.scores[item]

            return torch.from_numpy(user_features), torch.from_numpy(move_features), \
                   torch.tensor(score).float(), torch.tensor(score_norm)

        else:
            user_id, movie_id, score = self.test_samples[item]
            user_features = self.users[user_id - 1]
            move_features = self.movies[movie_id - 1]

            return torch.from_numpy(user_features), torch.from_numpy(move_features)

    def __len__(self):
        return len(self.train_samples) if self.mode == 'train' else len(self.test_samples)


class ScoreModule(nn.Module):
    def __init__(self, **kwargs):
        super(ScoreModule, self).__init__(**kwargs)
        self.user_features = nn.Sequential(nn.Linear(43, 64),
                                           nn.ReLU(),
                                           nn.Linear(64, 32),
                                           nn.BatchNorm1d(32),
                                           nn.ReLU(),
                                           nn.Linear(32, 8),
                                           nn.BatchNorm1d(8))

        self.movie_features = nn.Sequential(nn.Linear(19, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 32),
                                            nn.BatchNorm1d(32),
                                            nn.ReLU(),
                                            nn.Linear(32, 8),
                                            nn.BatchNorm1d(8))
        self.fc = nn.Linear(24, 1)

    def forward(self, users, movies):
        uf = self.user_features(users)

        mf = self.movie_features(movies)

        # scores = [torch.matmul(u, m) for u, m in zip(uf, mf)]
        # scores = torch.stack(scores)
        x = torch.cat([uf, mf, uf * mf], dim=1)
        x = self.fc(x)
        return x[:, 0]


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


def main(args):
    # 加载数据
    dataset = YxsDataset()
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

    # 加载网络
    net = ScoreModule()
    net.train()
    optimizer = optim.SGD(net.parameters(),
                          lr=1e-2, momentum=0.9, weight_decay=5e-6)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)

    epochs = 100
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.
        for users, movies, _, scores in tqdm(data_loader):
            scores_pred = net(users, movies)
            loss = F.mse_loss(scores_pred, scores)
            # optimizer.zero_grad()
            net.zero_grad()
            loss.backward()
            optimizer.step()
            # 当前轮的loss
            epoch_loss += loss.item() * users.size(0)

        # 更新lr
        lr_scheduler.step(epoch)

        epoch_loss = epoch_loss / len(data_loader.dataset)
        # 打印日志,保存权重
        print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, epochs, epoch_loss))

        # 预测精度
        if (epoch + 1) % 5 == 0:
            acc = accuracy(net, data_loader)
            print("epoch:{:03d} accuracy:{:.3f}".format(epoch + 1, acc))

        # 保存模型
        if args.output_dir and (epoch + 1) % 5 == 0:
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'args': args}
            torch.save(checkpoint,
                       os.path.join(args.output_dir, 'yxs79.{:03d}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output-dir', default='./', help='path where to save')
    # arguments = parser.parse_args(sys.argv[1:])
    #
    # main(arguments)
    cat_boost_train()
