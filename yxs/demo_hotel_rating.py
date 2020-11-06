# -*- coding: utf-8 -*-
"""
 @File    : demo_hotel_rating.py
 @Time    : 2020/11/5 上午11:37
 @Author  : yizuotian
 @Description    :
"""
import numpy as np
import pandas as pd
import torch
from transformers import AdamW
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


def main(csv_path):
    device = torch.device('cuda:5')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.to(device)
    train_data = pd.read_csv(csv_path, sep=',')
    review_np = train_data['review'].values
    rating_np = train_data['rating'].values.astype(np.int32) - 1  # 从0开始

    train(model, tokenizer, optimizer, review_np, rating_np, 4)

    evaluate(model, tokenizer, review_np, rating_np, 10)

    # 预测
    test_df = pd.read_csv('/sdb/tmp/open_dataset/Hotel_rating/test.csv')
    test_df['rating'] = inference(model, tokenizer, test_df['review'].values, device, 4)
    test_df[['id', 'rating']].to_csv('rst_hotel_rating.csv', header=None, index=None)


def train(model, tokenizer, optimizer, review_np, rating_np, epochs, device, batch_size=4):
    model.train()
    num = len(rating_np)
    for epoch in range(epochs):
        # 打乱
        ix = np.random.choice(num, num, False)
        review_np = review_np[ix]
        rating_np = rating_np[ix]

        for i in range(len(rating_np) // batch_size):
            text_batch = review_np[i * batch_size:(i + 1) * batch_size]
            encoding = tokenizer(list(text_batch), return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            rating_batch = rating_np[i * batch_size:(i + 1) * batch_size]
            labels = torch.tensor(rating_batch).unsqueeze(0).to(device)
            # 梯度更新
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # print(outputs)
            loss = outputs[0]
            loss.backward()
            optimizer.step()


def inference(model, tokenizer, text_np, device, batch_size):
    model.eval()
    logits = []
    for i in range(len(text_np) // batch_size):
        text_batch = text_np[i * batch_size:(i + 1) * batch_size]
        encoding = tokenizer(list(text_batch), return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        logits.append(outputs[0].cpu().detetch())
    mod = len(text_np) % batch_size
    if mod > 0:
        text_batch = text_np[-mod:]
        encoding = tokenizer(list(text_batch), return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits.append(outputs[0].cpu().detetch())

    logits = np.concatenate(logits, axis=0)
    labels = np.argmax(logits, axis=-1)
    return labels


def evaluate(model, tokenizer, text_np, labels, batch_size):
    predict_labels = inference(model, tokenizer, text_np, batch_size)
    return np.mean(predict_labels == labels)


if __name__ == '__main__':
    main()
