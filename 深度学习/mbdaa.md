a)结果记录,确定评估标准



| model    | k-fold(test,val) | 数据增广      | epochs    | test dataset acc   | 提交结果          | 精调stage | 备注       |
| -------- | :--------------- | --------- | --------- | ------------------ | ------------- | ------- | -------- |
| Resnet50 | 10,5             | a         | 5         | 0.855895197        | 0.8107585     | 5       |          |
| Resnet50 | 10,5             | a         | 10        | 0.868995633        | 0.81504107    | 5       |          |
| Resnet50 | 10,5             | b         | 5         | 0.8774617067833698 | 0.82123950000 | 5       |          |
| Resnet50 | 10,5             | b         | 5         | 0.8646288209606987 | 未提交           | 4       | 训练精度很好   |
| Resnet50 | 10,5             | b         | 3         | 0.8580786026200873 | 未提交           | 4       | 提前终止效果不好 |
| Resnet50 | 10,5             | c(去掉了旋转)  | 3         | 0.8580786026200873 | 未提交           | 4       |          |
| Resnet50 | 10,5             | c         | 10(需要更多轮) | 0.8755458515283843 | 未提交           | 4       |          |
| Resnet50 | 10,5             | d(变换程度减小) | 5         | 0.888646288209607  | 0.82884717000 | 4       |          |
| Resnet50 | 10,5             | e(增加白噪声)  |           | MemoryError        |               |         |          |



searchGrid

baseline:  sum_acc:0.8733624454148472,max_acc:0.8777292576419214



b) 固定模型和其它参数，searchGrid数据增广; rescale=1/255



| model    | 精调stage | epochs | k-fold (test,val) | 数据增广                          | test dataset acc                         | 提交结果 | 备注   |
| -------- | ------- | ------ | ----------------- | ----------------------------- | ---------------------------------------- | ---- | ---- |
| resnet50 | 4       | 5      | 10,5              | baseline                      | 0.8864628820960698 0.8668122270742358 0.8777292576419214 |      |      |
|          |         |        |                   | samplewise _center            | 0.7663755458515283                       |      |      |
|          |         |        |                   | samplewise_std_ normalization | 0.8034934497816594                       |      |      |
|          |         |        |                   | horizontal_flip               | 0.8733624454148472                       |      |      |
|          |         |        |                   | vertical_flip                 | 0.8602620087336245                       |      |      |
|          |         |        |                   | rotation_range                | 0.87117903930131                         |      |      |
|          |         |        |                   | width_shift_range             | 0.851528384279476                        |      |      |
|          |         |        |                   | height_shift_range            | 0.8646288209606987                       |      |      |
|          |         |        |                   | shear_range                   | 0.8668122270742358                       |      |      |
|          |         |        |                   | zoom_range                    |                                          |      |      |



测试集最终精度:sum_acc:0.8602620087336245,max_acc:0.8558951965065502
测试集最终精度:sum_acc:0.8602620087336245,max_acc:0.8668122270742358

测试集最终精度:sum_acc:0.8668122270742358,max_acc:0.8449781659388647
测试集最终精度:sum_acc:0.851528384279476,max_acc:0.8624454148471615
测试集最终精度:sum_acc:0.8668122270742358,max_acc:0.8646288209606987
测试集最终精度:sum_acc:0.851528384279476,max_acc:0.8471615720524017

测试集最终精度:sum_acc:0.8384279475982532,max_acc:0.8558951965065502
测试集最终精度:sum_acc:0.8580786026200873,max_acc:0.8537117903930131
测试集最终精度:sum_acc:0.8668122270742358,max_acc:0.8558951965065502
测试集最终精度:sum_acc:0.8558951965065502,max_acc:0.8537117903930131
测试集最终精度:sum_acc:0.8493449781659389,max_acc:0.8558951965065502

测试集最终精度:sum_acc:0.8820960698689956,max_acc:0.8799126637554585

===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098}

===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'samplewise_center': True}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'samplewise_center': True, 'samplewise_std_normalization': True}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'horizontal_flip': True}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'vertical_flip': True}

===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'rotation_range': 10}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'width_shift_range': 0.1}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'height_shift_range': 0.1}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'shear_range': 0.1}
===current epoch:1,,data_gen_args:{'rescale': 0.00392156862745098, 'zoom_range': 0.1}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098}

总结：epochs 1来看samplewise_center，horizontal_flip，height_shift_range 有少许提升





(keras) [root@localhost mbdaa]# grep "sum_acc" *.log
2018-10-13 20:43:16 测试集最终精度:sum_acc:0.8755458515283843,max_acc:0.8755458515283843

2018-10-13 21:05:18 测试集最终精度:sum_acc:0.8820960698689956,max_acc:0.8755458515283843
2018-10-13 21:32:17 测试集最终精度:sum_acc:0.8733624454148472,max_acc:0.8842794759825328
2018-10-13 22:04:47 测试集最终精度:sum_acc:0.868995633187773,max_acc:0.8842794759825328
2018-10-13 22:43:19 测试集最终精度:sum_acc:0.8777292576419214,max_acc:0.868995633187773

2018-10-13 23:28:30 测试集最终精度:sum_acc:0.8558951965065502,max_acc:0.8646288209606987
2018-10-14 00:20:54 测试集最终精度:sum_acc:0.8624454148471615,max_acc:0.8668122270742358
2018-10-14 01:21:48 测试集最终精度:sum_acc:0.8755458515283843,max_acc:0.868995633187773
2018-10-14 02:31:34 测试集最终精度:sum_acc:0.8646288209606987,max_acc:0.8755458515283843
2018-10-14 03:51:52 测试集最终精度:sum_acc:0.8646288209606987,max_acc:0.8733624454148472





2018-10-14 05:32:07 测试集最终精度:sum_acc:0.8755458515283843,max_acc:0.87117903930131

2018-10-14 07:25:55 测试集最终精度:sum_acc:0.8777292576419214,max_acc:0.8668122270742358







(keras) [root@localhost mbdaa]# grep "sum_acc" *.log
2018-10-14 10:22:45 测试集最终精度:sum_acc:0.8733624454148472,max_acc:0.87117903930131
2018-10-14 10:41:10 测试集最终精度:sum_acc:0.8755458515283843,max_acc:0.8799126637554585

2018-10-14 10:59:33 测试集最终精度:sum_acc:0.8733624454148472,max_acc:0.87117903930131
2018-10-14 11:17:55 测试集最终精度:sum_acc:0.8668122270742358,max_acc:0.8777292576419214
2018-10-14 11:36:15 测试集最终精度:sum_acc:0.868995633187773,max_acc:0.8668122270742358
2018-10-14 11:54:35 测试集最终精度:sum_acc:0.8733624454148472,max_acc:0.8558951965065502

2018-10-14 12:12:51 测试集最终精度:sum_acc:0.8646288209606987,max_acc:0.8624454148471615
2018-10-14 12:31:09 测试集最终精度:sum_acc:0.8820960698689956,max_acc:0.8842794759825328
2018-10-14 12:49:28 测试集最终精度:sum_acc:0.87117903930131,max_acc:0.8733624454148472
2018-10-14 13:07:49 测试集最终精度:sum_acc:0.8777292576419214,max_acc:0.8777292576419214
2018-10-14 13:26:04 测试集最终精度:sum_acc:0.8733624454148472,max_acc:0.868995633187773
(keras) [root@localhost mbdaa]# grep "=current" *.log
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'samplewise_center': True}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'samplewise_center': True, 'samplewise_std_normalization': True}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'horizontal_flip': True}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'vertical_flip': True}

===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'rotation_range': 10}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'width_shift_range': 0.1}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'height_shift_range': 0.1}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'shear_range': 0.1}
===current epoch:5,rescale:1.0,data_gen_args:{'rescale': 1.0, 'zoom_range': 0.1}
(keras) [root@localhost mbdaa]# 





(keras) [root@localhost mbdaa]# 
(keras) [root@localhost mbdaa]# grep "==current" *.log
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098}

===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'samplewise_center': True}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'samplewise_center': True, 'samplewise_std_normalization': True}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'horizontal_flip': True}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'vertical_flip': True}

===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'rotation_range': 10}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'width_shift_range': 0.1}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'height_shift_range': 0.1}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'shear_range': 0.1}
===current epoch:5,,data_gen_args:{'rescale': 0.00392156862745098, 'zoom_range': 0.1}

===current epoch:8,,data_gen_args:{'rescale': 0.00392156862745098}
===current epoch:8,,data_gen_args:{'rescale': 0.00392156862745098, 'samplewise_center': True}





c) 固定模型和其它参数，searchGrid数据增广; 

baseline1:rescale=1/255

baseline2:rescale=1/255,samplewise _center=True,samplewise_std_ normalization=True

后面都是基于baseline2

| model    | 精调stage | epochs | k-fold (test,val) | 数据增广               | test dataset acc   | 提交结果 | 备注   |
| -------- | ------- | ------ | ----------------- | ------------------ | ------------------ | ---- | ---- |
| resnet50 | 4       | 5      | 10,5              | baseline1          | 0.8995633187772926 |      |      |
|          |         |        |                   | baseline2          | 0.8842794759825328 |      |      |
|          |         |        |                   | horizontal_flip    | 0.8471615720524017 |      |      |
|          |         |        |                   | vertical_flip      | 0.8602620087336245 |      |      |
|          |         |        |                   | rotation_range     | 0.8733624454148472 |      |      |
|          |         |        |                   | width_shift_range  | 0.8755458515283843 |      |      |
|          |         |        |                   | height_shift_range | 0.8755458515283843 |      |      |
|          |         |        |                   | shear_range        | 0.8799126637554585 |      |      |




d) 固定模型和其它参数，searchGrid数据增广; rescale=1.0 (精准惨不忍睹)



| model    | 精调stage | epochs | k-fold (test,val) | 数据增广                          | test dataset acc   | 提交结果 | 备注   |
| -------- | ------- | ------ | ----------------- | ----------------------------- | ------------------ | ---- | ---- |
| resnet50 | 4       | 3      | 10,5              | baseline                      | 0.4781659388646288 |      |      |
|          |         |        |                   | samplewise_center             |                    |      |      |
|          |         |        |                   | samplewise_std_ normalization |                    |      |      |
|          |         |        |                   | horizontal_flip               |                    |      |      |
|          |         |        |                   | vertical_flip                 |                    |      |      |
|          |         |        |                   | rotation_range                |                    |      |      |
|          |         |        |                   | width_shift_range             |                    |      |      |
|          |         |        |                   | height_shift_range            |                    |      |      |
|          |         |        |                   | shear_range                   |                    |      |      |







经验总结

1. 精调更多层，训练精度更高(5个epoch 98%)，也更容易过拟合；
2. 做了数据增广，训练轮数需要更多，没有数据增广，数据轮数要相应减少
3. 做了数据增广，精调较少层时，训练精度上去比较慢(10个 epoch 88%)
4. ​



数据增广

a)

```python
data_gen_args = dict(rotation_range=40,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     rescale=rescale,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     validation_split=0.02,
                     fill_mode='wrap')
```

b)

```python
data_gen_args = dict(rescale=rescale)
```



c)

```
data_gen_args = dict(width_shift_range=0.2,
                     height_shift_range=0.2,
                     rescale=rescale,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='wrap')
```



d)

```python
data_gen_args = dict(width_shift_range=0.1,
                     height_shift_range=0.1,
                     rescale=rescale,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='wrap')
```



e) 增加白噪声（Traceback (most recent call last):  File "train.py", line 183, in <module>    main(sys.argv[1])  File "train.py", line 40, in main    kfold_train(model_name)  File "train.py", line 72, in kfold_train    datagen.fit(X_trainval)  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras_preprocessing/image.py", line 1215, in fit    sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]MemoryError）

```python
data_gen_args = dict(zca_whitening=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     rescale=rescale,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='wrap')
```





## 问题记录

1：gpu显存垃圾回收

2：测试集每次不一样

3：训练和测试验证的标准化问题

4：python后面越来越慢

5：半监督学习，加入的样本必须确保完全正确，错误样本严重影响精度

6：



nohup python grid_search.py > grid_search.·date "+%Y%m%d%H%M%S"·.log 2>&1 &



TODO：加入背景图片



代码备份

```
# 分割训练验证和测试
trainval_idx, test_idx = next(trainval_test_kfold.split(X=np.zeros(len(y)), y=np.argmax(y, axis=-1)))

f_trainval, l_trainval = file_list[trainval_idx], label_list[trainval_idx]
f_test, l_test = file_list[test_idx], label_list[test_idx]

X_test, y_test = data.load_img(f_test, l_test)  # 测试集
X_trainval, y_trainval = data.load_img(f_trainval, l_trainval)  # 测试集
```


```
# dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True),
# dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True,
#      horizontal_flip=True),
# dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True,
#      vertical_flip=True),
# dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True,
#      rotation_range=10),
dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True,
     width_shift_range=0.1),
dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True,
     height_shift_range=0.1),
dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True,
     shear_range=0.1),
dict(rescale=config.rescale, samplewise_center=True, samplewise_std_normalization=True, zoom_range=0.1)
```



```
"""dict(rescale=config.rescale, horizontal_flip=True),
dict(rescale=config.rescale, vertical_flip=True),
dict(rescale=config.rescale, rotation_range=10),
dict(rescale=config.rescale, width_shift_range=0.1),
dict(rescale=config.rescale, height_shift_range=0.1),
dict(rescale=config.rescale, shear_range=0.1),
dict(rescale=config.rescale, zoom_range=0.1)"""
```