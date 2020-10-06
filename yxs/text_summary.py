# -*- coding: utf-8 -*-
"""
 @File    : text_summary.py
 @Time    : 2020/10/4 上午9:33
 @Author  : yizuotian
 @Description    :
"""
import re

import jiagu
import macropodus
import pandas as pd
from pyhanlp import *
from textrank4zh import TextRank4Sentence


def filter_text(text):
    str_list = ['1、本文是[芥末堆网](//www.jiemodui.com)']
    for s in str_list:
        if text.__contains__(s):
            text = text[:text.index(s)]
            break
    re_tag = re.compile('([^!])\[(.*?)\]\(.*?\)')  # MD文字链接
    new_text = re.sub(re_tag, '\g<1>\g<2>', text)

    re_tag = re.compile('!\[.*?\]\(.*?\)')  # MD文字图像链接
    new_text = re.sub(re_tag, '', new_text)

    return new_text


def filter_rec(rec):
    return filter_text(rec['article'])


def summary_jiagu(rec):
    text = rec['article']
    return jiagu.summarize(text, 1)[0]


def summary_text_rank(rec):
    text = rec['article']
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    rst = list(tr4s.get_key_sentences(sentence_min_len=1))
    if len(rst) >= 1:
        return rst[0]['sentence']
    return text


def summary_hanlp(rec):
    text = rec['article']
    if text is None or len(text) == 0:
        print(rec)
    rst = list(HanLP.extractSummary(text, 1))
    if len(rst) >= 1:
        return rst[0]
    return text


def summary_macropodus(rec):
    text = rec['article']
    try:
        if text is None or len(text) == 0:
            print(rec)
        rst = macropodus.summarization(text, type_summarize='textrank')
        if len(rst) >= 1:
            return rst[0][1]

        return text
    except Exception as e:
        print(rec)
        return text


def evaluate():
    from sumeval.metrics.rouge import RougeCalculator

    rouge = RougeCalculator(stopwords=True, lang="zh")

    rouge_1 = rouge.rouge_n(
        summary="I went to the Mars from my living town.",
        references="I went to Mars",
        n=1)

    rouge_2 = rouge.rouge_n(
        summary="I went to the Mars from my living town.",
        references=["I went to Mars", "It's my living town"],
        n=2)

    rouge_l = rouge.rouge_l(
        summary="I went to the Mars from my living town.",
        references=["I went to Mars", "It's my living town"])

    # You need spaCy to calculate ROUGE-BE

    rouge_be = rouge.rouge_be(
        summary="I went to the Mars from my living town.",
        references=["I went to Mars", "It's my living town"])

    print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}".format(
        rouge_1, rouge_2, rouge_l, rouge_be
    ).replace(", ", "\n"))


def main():
    import time
    from pandarallel import pandarallel
    from tqdm import tqdm
    tqdm.pandas(desc='pandas bar')
    pandarallel.initialize(progress_bar=True)
    train = pd.read_csv('text_summary_final/train.csv')
    test = pd.read_csv('text_summary_final/test.csv', header=None, names=['article'])

    test['article'] = test.apply(filter_rec, axis=1)
    test.iloc[2815][
        'article'] = '![640.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/640.webp_1.jpg)![2.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/2.webp_.jpg)![3.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/3.webp_.jpg)![4.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/4.webp_.jpg)![6.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/6.webp_1.jpg)![7.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/7.webp_1.jpg)![8.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/8.webp_1.jpg)![9.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/9.webp_.jpg)![10.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/10.webp_.jpg)![11.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/11.webp_.jpg)![12.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/12.webp_.jpg)![13.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/13.webp_.jpg)![14.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/14.webp_.jpg)![15.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/15.webp_.jpg)![16.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/16.webp_.jpg)![17.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/17.webp_.jpg)![18.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/18.webp_.jpg)![19.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/19.webp_.jpg)![20.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/20.webp_.jpg)![21.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/21.webp_.jpg)![22.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/22.webp_.jpg)![23.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/23.webp_.jpg)![24.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/24.webp_.jpg)![25.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/25.webp_.jpg)![26.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/26.webp_.jpg)![27.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/27.webp_.jpg)![28.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/28.webp_.jpg)![29.webp](http://www.jingmeiti.com/wp-content/uploads/2016/07/29.webp_.jpg)'
    print(test.iloc[2815]['article'])

    test.to_csv('text_summary_final/test.filter.csv', header=None)

    # test['summary'] = test.parallel_apply(summary_jiagu, axis=1)
    # test[['summary']].to_csv('rst_text_summary.jiagu.csv', header=None)

    print('{} replace done !'.format(time.time()))
    # test['summary'] = test.progress_apply(summary_hanlp, axis=1)
    # test[['summary']].to_csv('rst_text_summary.hanlp.csv', header=None)

    # test['summary'] = test.progress_apply(summary_text_rank, axis=1)
    # test[['summary']].to_csv('rst_text_summary.text_rank.csv', header=None)

    test['summary'] = test.parallel_apply(summary_macropodus, axis=1)
    test[['summary']].to_csv('rst_text_summary.macropodus.lda.csv', header=None)


if __name__ == '__main__':
    main()
