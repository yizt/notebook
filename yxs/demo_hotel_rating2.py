# -*- coding: utf-8 -*-
"""
 @File    : demo_hotel_rating2.py
 @Time    : 2020/11/6 上午10:14
 @Author  : yizuotian
 @Description    : 使用transformers.Trainer训练
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, TrainingArguments, Trainer


class HotelDataSet(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # encodings.key(): ['input_ids', 'token_type_ids', 'attention_mask']
        item = {key: val[idx] for key, val in self.encodings.items()}  # 取指定的那个元素
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class HotelTestDataSet(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # encodings.key(): ['input_ids', 'token_type_ids', 'attention_mask']
        item = {key: val[idx] for key, val in self.encodings.items()}  # 取指定的那个元素
        return item

    def __len__(self):
        return len(self.labels)


def main(args):
    train_csv_path = os.path.join(args.data_root, 'train.csv')
    train_data = pd.read_csv(train_csv_path, sep=',')
    review_np = train_data['review'].values
    rating_np = train_data['rating'].values.astype(np.long) - 1  # 从0开始
    # 分割训练和验证
    train_texts, val_texts, train_labels, val_labels = train_test_split(review_np, rating_np, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

    train_dataset = HotelDataSet(train_encodings, train_labels)
    val_dataset = HotelDataSet(val_encodings, val_labels)

    #
    # device = torch.device('cuda:5')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    # model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=6,  # batch size per device during training
        per_device_eval_batch_size=6,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()

    # 预测
    test_csv_path = os.path.join(args.data_root, 'test.csv')
    test_df = pd.read_csv(test_csv_path, sep=',')
    test_encoding = tokenizer(list(test_df['review'].values), truncation=True, padding=True)
    test_dataset = HotelTestDataSet(test_encoding)
    outputs = trainer.predict(test_dataset)

    labels = np.argmax(outputs.predictions, axis=-1) + 1  # label还原
    test_df['rating'] = labels
    test_df[['id', 'rating']].to_csv('./rst_hotel_rating.csv', header=None, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./Hotel_rating')
    arguments = parser.parse_args(sys.argv[1:])
    main(arguments)
