# -*- coding: utf-8 -*-
"""
 @File    : leave_office.py
 @Time    : 2020/10/2 上午7:43
 @Author  : yizuotian
 @Description    :
"""
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold


def train_one(X_train, X_test, y_train, y_test, cat_feature_indices):
    model = cb.CatBoostClassifier(iterations=1000,
                                  learning_rate=0.1,
                                  od_type='Iter',
                                  model_size_reg=3,
                                  depth=10,
                                  l2_leaf_reg=3)
    model.fit(X_train, y_train, cat_features=cat_feature_indices, eval_set=(X_test, y_test))
    print('train acc:{} , test acc:{}'.format(model.score(X_train, y_train),
                                              model.score(X_test, y_test)))
    return model


def eval_model(model_list, X, y, test):
    for i, model in enumerate(model_list):
        print('model {:02d} acc:{:0.3f}'.format(i, model.score(X, y)))

    # 所有模型一块预测
    predict_all = [model.predict_proba(X) for model in model_list]
    predict_all = np.stack(predict_all, axis=0)

    # 最大概率
    predict_max = np.argmax(np.max(predict_all, axis=0), axis=-1)
    print(predict_all.shape, y.shape, predict_max.shape)
    print('model max acc:{:0.3f}'.format(np.mean(predict_max == y)))
    # 平均概率
    predict_mean = np.argmax(np.mean(predict_all, axis=0), axis=-1)
    print('model mean acc:{:0.3f}'.format(np.mean(predict_mean == y)))
    # 投票
    predict_more = np.argmax(np.sum(predict_all == np.max(predict_all, axis=-1, keepdims=True),
                                    axis=0), axis=-1)
    print('model more acc:{:0.3f}'.format(np.mean(predict_more == y)))

    # 所有模型一块预测
    predict_all = [model.predict_proba(test) for model in model_list]
    predict_all = np.stack(predict_all, axis=0)
    # 最大概率
    predict_max = np.argmax(np.max(predict_all, axis=0), axis=-1)

    # 平均概率
    predict_mean = np.argmax(np.mean(predict_all, axis=0), axis=-1)

    # 投票
    predict_more = np.argmax(np.sum(predict_all == np.max(predict_all, axis=-1, keepdims=True),
                                    axis=0), axis=-1)

    return predict_max, predict_mean, predict_more


def main():
    """
    n_splits=4, n_repeats=10, random_state=100
    model max acc:0.984
    model mean acc:0.993
    model more acc:0.995

    n_splits=4, n_repeats=5, random_state=100
    model max acc:0.986
    model mean acc:0.994
    model more acc:0.995

    n_splits=5, n_repeats=10, random_state=100
    model max acc:0.985
    model mean acc:0.993
    model more acc:0.995

    n_splits=3, n_repeats=5, random_state=100
    model max acc:0.985
    model mean acc:0.992
    model more acc:0.994

    n_splits=5, n_repeats=5, random_state=100
    model max acc:0.987
    model mean acc:0.993
    model more acc:0.994

    n_splits=5, n_repeats=2, random_state=100
    model max acc:0.988
    model mean acc:0.994
    model more acc:0.995

    n_splits=4, n_repeats=2, random_state=100
    model max acc:0.989
    model mean acc:0.994
    model more acc:0.995
    n_splits=4, n_repeats=1, random_state=100
    model max acc:0.989
    model mean acc:0.994
    model more acc:0.994

    n_splits=4, n_repeats=3, random_state=100
    model max acc:0.988
    model mean acc:0.994
    model more acc:0.995
#########depth=10
    model max acc:0.988
    model mean acc:0.994
    model more acc:0.995

    n_splits=5, n_repeats=3, random_state=100
    model max acc:0.989
    model mean acc:0.996
    model more acc:0.997

    n_splits=6, n_repeats=3, random_state=100
    model max acc:0.989
    model mean acc:0.996
    model more acc:0.997

    :return:
    """
    train = pd.read_csv('LeaveOffice/train.csv')
    test = pd.read_csv('LeaveOffice/test.csv')

    y = train['left']
    X = train.drop(columns=['left', 'id'])

    test_x = test.drop(columns=['id'])

    cat_feature_indices = [i for i, x in enumerate(X.columns) if x in ['sales', 'salary']]

    rkf = RepeatedKFold(n_splits=6, n_repeats=3, random_state=100)
    model_list = []
    for train_index, test_index in rkf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_list.append(train_one(X_train, X_test, y_train, y_test, cat_feature_indices))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, l2_leaf_reg=3)
    #
    # model.fit(X_train, y_train, cat_features=cat_feature_indices, eval_set=(X_test, y_test))
    #
    # print(confusion_matrix(y, model.predict(X)))

    p_max, p_mean,p_more = eval_model(model_list, X, y, test_x)

    test['left'] = p_max
    test[['id', 'left']].to_csv('rst_leave_office.max.csv',
                                header=None,
                                encoding='utf-8',
                                index=None)

    test['left'] = p_mean
    test[['id', 'left']].to_csv('rst_leave_office.mean.csv',
                                header=None,
                                encoding='utf-8',
                                index=None)

    test['left'] = p_more
    test[['id', 'left']].to_csv('rst_leave_office.more.csv',
                                header=None,
                                encoding='utf-8',
                                index=None)


if __name__ == '__main__':
    main()
