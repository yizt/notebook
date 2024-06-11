# -*- coding: utf-8 -*-
"""
 @File    : svm.py
 @Time    : 2020/7/10 下午11:09
 @Author  : yizuotian
 @Description    :
"""
import codecs
import numpy as np


def to_category(val, num_classes):
    val_list = [0] * num_classes
    val_list[int(val)] = 1
    return val_list


def load_train(csv_file):
    """
    #id,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal

    :return:
    """
    with codecs.open(csv_file) as f:
        lines = f.readlines()
    record_list = []
    target_list = []
    for line in lines[1:]:
        id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, \
        oldpeak, slope, ca, thal, label = line.split(',')

        record = [age, sex, trestbps, chol, fbs, restecg, thalach, oldpeak,
                  slope, ca, thal] + to_category(cp, 4) + to_category(exang, 3)

        record = list(map(float, record))
        record_list.append(record)
        target_list.append(int(label))

    return record_list, target_list


def load_test(csv_file):
    """
    #id,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal

    :return:
    """
    with codecs.open(csv_file) as f:
        lines = f.readlines()
    record_list = []
    for line in lines[1:]:
        id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, \
        oldpeak, slope, ca, thal = line.split(',')

        record = [age, sex, trestbps, chol, fbs, restecg, thalach, oldpeak,
                  slope, ca, thal] + to_category(cp, 4) + to_category(exang, 3)

        record = list(map(float, record))
        record_list.append(record)

    return record_list


if __name__ == '__main__':
    train_data, target = load_train('/Users/admin/Downloads/yanxishe_77/train.csv')

    from sklearn.utils import Bunch

    from sklearn import svm,tree
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score
    # from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler, Normalizer,MinMaxScaler
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder

    clf = svm.SVC(kernel='rbf', C=1)
    # clf = svm.SVC(kernel='poly', class_weight='balanced')
    clf = tree.DecisionTreeClassifier(max_depth=3)
    tree.export_graphviz
    # clf = GradientBoostingClassifier()
    model = SelectFromModel(estimator=tree.DecisionTreeClassifier())
    model.fit(train_data, target)
    train_data = model.transform(train_data)
    ss = StandardScaler().fit(train_data)
    train_data = ss.transform(train_data)
    clf.fit(train_data, target)

    # print(clf.feature_importances_)
    scores = cross_val_score(clf, train_data, target, cv=5)
    print(np.mean(scores), scores)

    test_data = load_test('/Users/admin/Downloads/yanxishe_77/test.csv')
    test_data = model.transform(test_data)
    test_data = ss.transform(test_data)
    ys = clf.predict(test_data)
    with codecs.open('answer.csv', mode='w', encoding='utf-8') as w:
        for i, y in enumerate(ys):
            w.write('{},{}\n'.format(i, y))
