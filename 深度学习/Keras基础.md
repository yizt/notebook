[TOC]



## 模型

### 序贯模型

```python
def get_model(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu',input_shape=input_shape))
    model.add(Conv2D(32, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(64, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
    return model
```



### 函数式模型

```python
def get_model(input_shape):
    inputs = keras.layers.Input(shape=input_shape)
    
    # first conv
    x = Conv2D(32, (3, 3), padding='same',activation='relu')(inputs)
    x = Conv2D(32, (3, 3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # second conv
    x = Conv2D(64, (3, 3), padding='same',activation='relu')(x)
    x = Conv2D(64, (3, 3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # fc
    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes,activation='softmax')(x)
    
    model = keras.Model(inputs=inputs,outputs=x)
    return model
```





## 层

### 自定义层

要定制自己的层，你需要实现下面三个方法

- `build(input_shape)`：这是定义权重的方法，可训练的权应该在这里被加入列表``self.trainable_weights`中。其他的属性还包括`self.non_trainabe_weights`（列表）和`self.updates`（需要更新的形如（tensor, new_tensor）的tuple的列表）。你可以参考`BatchNormalization`层的实现来学习如何使用上面两个属性。这个方法必须设置`self.built = True`，可通过调用`super([layer],self).build()`实现
- `call(x)`：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心`call`的第一个参数：输入张量
- `get_output_shape_for(input_shape)`：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
    # 这是定义权重的方法，可训练的权应该在这里被加入列表
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                initializer='random_uniform',
                                trainable=True)
        super(MyLayer, self).build()  # be sure you call this somewhere! 

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] + self.output_dim)
```





## 损失函数

目标函数，或称损失函数，是编译一个模型必须的两个参数之一：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

可以通过传递预定义目标函数名字指定目标函数，也可以传递一个Theano/TensroFlow的符号函数作为目标函数，该函数对每个数据点应该只返回一个标量值，并以下列两个参数为参数：

- y_true：真实的数据标签，Theano/TensorFlow张量
- y_pred：预测值，与y_true相同shape的Theano/TensorFlow张量

```python
from keras import losses
model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

### 可用的目标函数

- mean_squared_error或mse
- mean_absolute_error或mae
- mean_absolute_percentage_error或mape
- mean_squared_logarithmic_error或msle
- squared_hinge
- hinge
- categorical_hinge
- binary_crossentropy（亦称作对数损失，logloss）
- logcosh
- categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如`(nb_samples, nb_classes)`的二值序列
- sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：`np.expand_dims(y,-1)`
- kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
- poisson：即`(predictions - targets * log(predictions))`的均值
- cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数

**注意**: 当使用"categorical_crossentropy"作为目标函数时,标签应该为多类模式,即one-hot编码的向量,而不是单个数值. 可以使用工具中的`to_categorical`函数完成该转换.示例如下:

```python
from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)
```



### add_loss定义loss层



```python
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
```





```python
# Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)
```





## 优化器

​        参数 *clipnorm* 和 *clipvalue* 是所有优化器都可以使用的参数,用于对梯度进行裁剪

```python
from keras import optimizers
#在进行梯度下降算法过程中，梯度值可能会较大，通过控制clipnorm可以实现梯度的剪裁
#使其2范数不超过clipnorm所设定的值
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```



```python
from keras import optimizers
#在进行梯度下降算法过程中，梯度值可能会较大，通过控制clipnorm可以实现梯度的剪裁
#使其2范数不超过clipvalue所设定的值
#比如-1，-2，-0.6等会被修剪为会被修剪为-0.5，而-0.4，0.3等不变
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```



### SGD

```python
keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
```

- lr：大或等于0的浮点数，学习率
- momentum：大或等于0的浮点数，动量参数
- decay：大或等于0的浮点数，每次更新后的学习率衰减值
- nesterov：布尔值，确定是否使用Nesterov动量



### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
```

除学习率可调整外，建议保持优化器的其他默认参数不变

该优化器通常是面对递归神经网络时的一个良好选择

- lr：大或等于0的浮点数，学习率
- rho：大或等于0的浮点数
- epsilon：大或等于0的小浮点数，防止除0错误
- decay：大或等于0的浮点数，每次更新后的学习率衰减值





### Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

该优化器的默认值来源于参考文献

- lr：大或等于0的浮点数，学习率
- beta_1/beta_2：浮点数， 0<beta<1，通常很接近1
- epsilon：大或等于0的小浮点数，防止除0错误
- decay：大或等于0的浮点数，每次更新后的学习率衰减值







## 评估器

性能评估模块提供了一系列用于模型性能评估的函数,这些函数在模型编译时由`metrics`关键字设置

性能评估函数类似与**目标函数**, 只不过该性能的评估结果讲不会用于训练.

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])

from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

- y_true:真实标签,theano/tensorflow张量
- y_pred:预测值, 与y_true形式相同的theano/tensorflow张量



### 可用预定义张量

除fbeta_score额外拥有默认参数beta=1外,其他各个性能指标的参数均为y_true和y_pred

- binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率
- categorical_accuracy:对多分类问题,计算再所有预测值上的平均正确率
- sparse_categorical_accuracy:与`categorical_accuracy`相同,在对稀疏的目标值预测时有用
- top_k_categorical_accracy: 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
- sparse_top_k_categorical_accuracy：与top_k_categorical_accracy作用相同，但适用于稀疏情况

```python
keras.metrics.binary_accuracy(y_true, y_pred)
```



## 回调函数Callbacks

​        回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。通过传递回调函数列表到模型的`.fit()`中，即可在给定的训练阶段调用该函数集中的函数。

【Tips】虽然我们称之为回调“函数”，但事实上Keras的**回调函数** **是一个类**，回调函数只是习惯性称呼

### Callback

```
keras.callbacks.Callback()

```

这是回调函数的抽象类，定义新的回调函数必须继承自该类

#### 类属性

- params：字典，训练参数集（如信息显示方法verbosity，batch大小，epoch数）
- model：`keras.models.Model`对象，为正在训练的模型的引用

回调函数以字典`logs`为参数，该字典包含了一系列与当前batch或epoch相关的信息。

目前，模型的`.fit()`中有下列参数会被记录到`logs`中：

- 在每个epoch的结尾处（on_epoch_end），`logs`将包含训练的正确率和误差，`acc`和`loss`，如果指定了验证集，还会包含验证集正确率和误差`val_acc)`和`val_loss`，`val_acc`还额外需要在`.compile`中启用`metrics=['accuracy']`。
- 在每个batch的开始处（on_batch_begin）：`logs`包含`size`，即当前batch的样本数
- 在每个batch的结尾处（on_batch_end）：`logs`包含`loss`，若启用`accuracy`则还包含`acc`



### BaseLogger

```python
keras.callbacks.BaseLogger()
```

该回调函数用来对每个epoch累加`metrics`指定的监视指标的epoch平均值

该回调函数在每个Keras模型中都会被自动调用

------

### ProgbarLogger

```python
keras.callbacks.ProgbarLogger()
```

该回调函数用来将`metrics`指定的监视指标输出到标准输出上

------

### History

```python
keras.callbacks.History()
```

该回调函数在Keras模型上会被自动调用，`History`对象即为`fit`方法的返回值

------

### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

该回调函数将在每个epoch后保存模型到`filepath`

`filepath`可以是格式化的字符串，里面的占位符将会被`epoch`值和传入`on_epoch_end`的`logs`关键字所填入

例如，`filepath`若为`weights.{epoch:02d-{val_loss:.2f}}.hdf5`，则会生成对应epoch和验证集loss的多个文件。

#### 参数

- filename：字符串，保存模型的路径
- monitor：需要监视的值
- verbose：信息展示模式，0或1
- save_best_only：当设置为`True`时，将只保存在验证集上性能最好的模型
- mode：‘auto’，‘min’，‘max’之一，在`save_best_only=True`时决定性能最佳模型的评判准则，例如，当监测值为`val_acc`时，模式应为`max`，当检测值为`val_loss`时，模式应为`min`。在`auto`模式下，评价准则由被监测值的名字自动推断。
- save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
- period：CheckPoint之间的间隔的epoch数



### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
```

当监测值不再改善时，该回调函数将中止训练

#### 参数

- monitor：需要监视的量
- patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过`patience`个epoch后停止训练。
- verbose：信息展示模式
- mode：‘auto’，‘min’，‘max’之一，在`min`模式下，如果检测值停止下降则中止训练。在`max`模式下，当检测值不再上升则停止训练。

------

### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000')
```

该回调函数用于向服务器发送事件流，该回调函数需要`requests`库

#### 参数

- root：该参数为根url，回调函数将在每个epoch后把产生的事件流发送到该地址，事件将被发往`root + '/publish/epoch/end/'`。发送方法为HTTP POST，其`data`字段的数据是按JSON格式编码的事件字典。

------

### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule)
```

该回调函数是学习率调度器

#### 参数

- schedule：函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）

------

### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
```

该回调函数是一个可视化的展示器

TensorBoard是TensorFlow提供的可视化工具，该回调函数将日志信息写入TensorBorad，使得你可以动态的观察训练和测试指标的图像以及不同层的激活值直方图。

如果已经通过pip安装了TensorFlow，我们可通过下面的命令启动TensorBoard：

```shell
tensorboard --logdir=/full_path_to_your_logs
```

更多的参考信息，请点击[这里](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

#### 参数

- log_dir：保存日志文件的地址，该文件将被TensorBoard解析以用于可视化
- histogram_freq：计算各个层激活值直方图的频率（每多少个epoch计算一次），如果设置为0则不计算。
- write_graph: 是否在Tensorboard上可视化图，当设为True时，log文件可能会很大
- write_images: 是否将模型权重以图片的形式可视化
- embeddings_freq: 依据该频率(以epoch为单位)筛选保存的embedding层
- embeddings_layer_names:要观察的层名称的列表，若设置为None或空列表，则所有embedding层都将被观察。
- embeddings_metadata: 字典，将层名称映射为包含该embedding层元数据的文件名，参考[这里](https://keras.io/https__://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)获得元数据文件格式的细节。如果所有的embedding层都使用相同的元数据文件，则可传递字符串。

------

### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

当评价指标不在提升时，减少学习率

当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在`patience`个epoch中看不到模型性能提升，则减少学习率

#### 参数

- monitor：被监测的量
- factor：每次减少学习率的因子，学习率将以`lr = lr*factor`的形式被减少
- patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
- mode：‘auto’，‘min’，‘max’之一，在`min`模式下，如果检测值触发学习率减少。在`max`模式下，当检测值不再上升则触发学习率减少。
- epsilon：阈值，用来确定是否进入检测值的“平原区”
- cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
- min_lr：学习率的下限

### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

将epoch的训练结果保存在csv文件中，支持所有可被转换为string的值，包括1D的可迭代数值如np.ndarray.

#### 参数

- fiename：保存的csv文件名，如`run/log.csv`
- separator：字符串，csv分隔符
- append：默认为False，为True时csv文件如果存在则继续写入，为False时总是覆盖csv文件

### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

用于创建简单的callback的callback类

该callback的匿名函数将会在适当的时候调用，注意，该回调函数假定了一些位置参数`on_eopoch_begin`和`on_epoch_end`假定输入的参数是`epoch, logs`. `on_batch_begin`和`on_batch_end`假定输入的参数是`batch, logs`，`on_train_begin`和`on_train_end`假定输入的参数是`logs`

#### 参数

- on_epoch_begin: 在每个epoch开始时调用
- on_epoch_end: 在每个epoch结束时调用
- on_batch_begin: 在每个batch开始时调用
- on_batch_end: 在每个batch结束时调用
- on_train_begin: 在训练开始时调用
- on_train_end: 在训练结束时调用

#### 示例

```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# Plot the loss after every epoch.
import numpy as np
import matplotlib.pyplot as plt
plot_loss_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
                      logs['loss']))

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
    p.terminate() for p in processes if p.is_alive()])

model.fit(...,
      callbacks=[batch_print_callback,
         plot_loss_callback,
         cleanup_callback])
```

### 编写自己的回调函数

我们可以通过继承`keras.callbacks.Callback`编写自己的回调函数，回调函数通过类成员`self.model`访问访问，该成员是模型的一个引用。

这里是一个简单的保存每个batch的loss的回调函数：

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

#### 例子：记录损失函数的历史数据

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print history.losses
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
```

#### 例子：模型检查点

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```



## backend

set_value

​           使用一个numpy数组给一个tensor变量赋值

```python
def set_value(x, value):
  """Sets the value of a variable, from a Numpy array.

  Arguments:
      x: Tensor to set to a new value.
      value: Value to set the tensor to, as a Numpy array
          (of the same shape).
  """
```



## 预训练模型

下载地址：https://github.com/fchollet/deep-learning-models/releases/



参考：[Keras中文手册](http://keras-cn.readthedocs.io/en/latest/)



## 图片预处理

```python
def gen_data(image_infos, batch_size=1):
    """
    生成器
    :param image_infos:
    :param batch_size:
    :return:
    """
    length = len(image_infos)
    while True:
        batch_image_infos = [image_infos[i] for i in np.random.randint(0, length, size=batch_size)]
        x, y = mask_img(batch_image_infos)
        yield x, y
 
m.fit_generator(generator=gen_data(train_img_infos, batch_size),
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=get_call_back(backbone_model),
                    initial_epoch=initial_epoch)
```





## 常见错误

1. tf.where(x>1,x,1);   tf.where中，x,y必须维度一致
2. tf.minimum(tf.shape(positive_indices)[0], train_anchors_num * positive_ratios); 数据类型必须一致

## 总结

1) tf数据类型：不同数据类型不能操作

2) Model多输入多输出,inputs，outputs都是list; 也可以是字典类型;Input层名字对应字典的key,如：

```python
	   inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)
    
input_data = Input(name='the_input', shape=input_shape, dtype='float32') 
labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
```



3) Layer多输入多输出，input_shape是list(tuple), compute_output_shape返回的也是list(tuple)；对于多输出call返回必须是列表，不能是tuple

4) 可以自定义loss层，直接使用`model.add_loss(loss)` ;例如：

```python
loss_names = ["rpn_bbox_loss", "rpn_class_loss"]  # , "rpn_bbox_loss",rpn_class_loss
    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer.output in keras_model.losses:
            continue
        loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.add_loss(loss)
```

5) 可以直接增加统计指标，使用`model.metrics_names.append`和`keras_model.metrics_tensors.append`  ，例如

```python
    layer = keras_model.get_layer('rpn_target')
    keras_model.metrics_names.append('gt_num')
    keras_model.metrics_tensors.append(layer.output[3])
```

​       <font color=#ff0000 >注意指标必须是float类型</font>

6) tf 调试,使用如下样例代码

```python
    from tensorflow.python import debug as tf_debug
    import keras.backend as K

    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    K.set_session(sess)
```

a) 在定义Tensor张量时，增加name属性

b) pt tensor (可以使用name名字检索)

c) ls -n Node ,按照节点**列出涉及节点创建的python源文件** ; tf报错堆栈,有时不会展示出错的源代码位置

参考：[Debugging 调试Tensorflow 程序](https://blog.csdn.net/u010312436/article/details/78656723/)

其它:使用tf.Print()和tf.Assert()打印变量

7) Input除了使用shape参数，也可以使用batch_shape,比shape长度大1；tuple第一个元素是batch_size大小

8) 孪生网络共享部分层的网络参数，如何设计？keras共享参数就是共享节点(层)；不共享参数，就不共享节点(层)

9) loss损失函数，如果有使用到最后一层的weights；通过将loss函数定义到用到weight的层中来实现；参考：https://github.com/LCorleone/Various-loss-function-in-face-recognition/blob/master/utils/utils_func.py

10) keras如下方式手工更新weights，初始权重; 但是无法在训练中手工更新权重：https://github.com/keras-team/keras/issues/11143

- `layer.set_weights(weights)`: sets the weights of the layer from a list of Numpy arrays (with the same shapes as the output of `get_weights`).


​        使用tf.add_to_collection(K.tf.GraphKeys.UPDATE_OPS, weights_update_op) 将修改权重的操作增加到计算图中.

​        如docface+的loss函数：

```python
weights_update_op_1 = K.tf.scatter_mul(self.kernel, labels, 1.0 - self.alpha)  # w = (1-a)*w
        weights_update_op_2 = K.tf.scatter_sub(self.kernel, labels, w_batch)  # w = w + a* w_batch

        with K.tf.control_dependencies([weights_update_op_1, weights_update_op_2]):
            weights_update_op = K.tf.assign(self.kernel,
                                            K.tf.nn.l2_normalize(self.kernel, axis=1))  # w_j^* 归一化
        # 加到Graph图中
        K.tf.add_to_collection(K.tf.GraphKeys.UPDATE_OPS, weights_update_op)
        # 最后返回损失函数
        return K.categorical_crossentropy(y_true, cosine_m * self.scale, from_logits=True)
```



11) keras 获取中间某一层的输出; 使用`K.function` ;

```python
fun = K.function([model.input], [model.layers[-2].output])
features = fun([new_imgs])[0]
```

​    参考：https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer



12) keras m.load_weights(pretrained_weights, by_name=True); 其中的layer一定要有name;切记



13) K.sparse_categorical_crossentropy 与keras.losses.sparse_categorical_crossentropy 目前效果有差别，不清楚为什么？前者训练不收敛？ 如下方式包装一下也不收敛，应当是keras本身bug吧

```python
    def my_sparse(y_true,y_pred):
        return keras.losses.sparse_categorical_crossentropy(y_true,y_pred)
```



14) 如果需要在训练的每一个step之后输出一些张量，可以讲张量保存到一个不训练的权重中；如下例子，说明如何在每个step之后获取预测的结果

```python
   def build(self, input_shape):
   		self.y_pred = self.add_weight(name='pred',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=False)
   def loss(self, y_true, y_pred):
        # 首先将预测值保持到权重中
        pred_assign_op = K.tf.assign(self.y_pred,
                                     y_pred,
                                     name='assign_pred')
        with K.tf.control_dependencies([pred_assign_op]):
            y_true = y_true[:, 0]  # 非常重要，默认是二维的
            y_true_mask = K.one_hot(K.tf.cast(y_true, dtype='int32'), self.output_dim)
            cosine_m = y_pred - y_true_mask * self.margin  # cosine-m
            losses = K.sparse_categorical_crossentropy(target=y_true,
                                                       output=cosine_m * self.scale,
                                                       from_logits=True)

        return losses
   
```

 

最后获取在callback中获取

```python
    def on_batch_end(self, batch, logs=None):
        layer = self.model.layers[-1]
        trained_weights, current_trained_labels, y_pred = layer.get_weights()[:3]
```



15：度量的值必须是float32类型

```
2/3 [===================>..........] - ETA: 4s - loss: 2.2542 - rpn_bbox_loss: 0.7040 - rpn_class_loss: 1.5502 - gt_num: 5.0000 - positive_anchor_num: 65.0000 - miss_match_gt_num: 2.7500Traceback (most recent call last):
  File "train.py", line 67, in <module>
    verbose=1)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/training.py", line 1426, in fit_generator
    initial_epoch=initial_epoch)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/training_generator.py", line 229, in fit_generator
    callbacks.on_epoch_end(epoch, epoch_logs)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/callbacks.py", line 77, in on_epoch_end
    callback.on_epoch_end(epoch, logs)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/callbacks.py", line 336, in on_epoch_end
    self.progbar.update(self.seen, self.log_values)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/utils/generic_utils.py", line 338, in update
    self._values[k][0] += v * (current - self._seen_so_far)
TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
```



16: layers的输入，必须是Input或者其他Layer的输出项，不能是做张量操作后的项，不然报错：`'NoneType' object has no attribute '_inbound_nodes'`

如下代码会报错：

```python
anchors = ClipBoxes(name='clip')([anchors, input_image_meta[:, 7:11]])
```

需要改为

```python
wins = Lambda(lambda x: x[:, 7:11])(input_image_meta)
anchors = ClipBoxes(name='clip')([anchors, wins])
```



报错如下：

```
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/network.py", line 1353, in build_map
    node_index, tensor_index)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/network.py", line 1325, in build_map
    node = layer._inbound_nodes[node_index]
AttributeError: 'NoneType' object has no attribute '_inbound_nodes'
```



17：tf.argmax需要考虑维度为零的情况，例如: GT 为0时

```python
    # anchors_iou_argmax = tf.argmax(iou, axis=0)  # 每个anchor最大iou对应的GT 索引 [n]
    anchors_iou_argmax = tf.cond(   # 需要考虑GT个数为0的情况
        tf.greater(tf.shape(iou)[0], 0),
        true_fn = lambda: tf.argmax(iou, axis=0),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
```



18：compute_output_shape函数没有作用；报`list index out of range`

```shell
Traceback (most recent call last):
  File "second_stage_retina_gt_stat.py", line 214, in <module>
    main()
  File "second_stage_retina_gt_stat.py", line 153, in main
    config.SECOND_STRIDE)
  File "second_stage_retina_gt_stat.py", line 126, in second_stage_retina_gt_net
    [input_second_boxes, input_second_class_ids, second_anchors, rois, second_windows, second_scales])
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/base_layer.py", line 497, in __call__
    arguments=user_kwargs)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/base_layer.py", line 565, in _add_inbound_node
    output_tensors[i]._keras_shape = output_shapes[i]
IndexError: list index out of range

```



原因：必须所有input_shape都不为None，compute_output_shape才有作用；如果输入中有标量，且Input中指定的是shape,那么input_shape中就有为None的，

```python
            if all([s is not None
                    for s in to_list(input_shape)]):
                output_shape = self.compute_output_shape(input_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None
```

解决办法：Input使用batch_shape初始化;如

```python
	input_image_meta = Input(batch_shape=(batch_size, 12))
    input_class_ids = Input(batch_shape=(batch_size, max_gt_num, 2))
```



19：Tensorflow: map_fn does not work in Graph mode during training, in Eager mode it does

```shell
    self._session._session, self._handle, args, status, None)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py", line 519, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.NotFoundError: Resource __per_step_8/_tensor_arraysrpn2proposals/map/TensorArray_2_17/N10tensorflow11TensorArrayE does not exist.
	 [[Node: training/SGD/gradients/rpn2proposals/map/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3 = TensorArrayGradV3[_class=["loc:@train...yScatterV3"], source="training/SGD/gradients", _device="/job:localhost/replica:0/task:0/device:CPU:0"](rpn2proposals/map/TensorArray_2/_2645, rpn2proposals/map/while/Exit_2/_2665)]]
	 [[Node: rpn2proposals/map_1/TensorArrayUnstack/range/_2906 = _HostSend[T=DT_INT32, client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_10460_rpn2proposals/map_1/TensorArrayUnstack/range", _device="/job:localhost/replica:0/task:0/device:GPU:0"](rpn2proposals/map_1/TensorArrayUnstack/range)]]
```

参考;https://stackoverflow.com/questions/52187269/tensorflow-map-fn-does-not-work-in-graph-mode-during-training-in-eager-mode-it



20：

```shell
1272/2505 [==============>...............] - ETA: 3:39 - loss: 2.1422 - rpn_bbox_loss: 0.8695 - rpn_class_loss: 0.2466 - rcnn_bbox_loss: 0.4770 - rcnn_class_loss: 0.5491 - gt_num: 3.1545 - positive_anchor_num: 16.7732 - miss_match_gt_num: 0.0000e+00 - gt_match_min_iou: 0.5314 - rcnn_miss_match_gt_num: 1.17102019-03-05 17:09:47.005883: W tensorflow/core/framework/op_kernel.cc:1318] OP_REQUIRES failed at tensor_array_ops.cc:497 : Invalid argument: TensorArray rcnn_target/map/TensorArray_5_175757@training/SGD/gradients: Could not read from TensorArray index 1.  Furthermore, the element shape is not fully defined: <unknown>.  It is possible you are working with a resizeable TensorArray and stop_gradients is not allowing the gradients to be written.  If you set the full element_shape property on the forward TensorArray, the proper all-zeros tensor will be returned instead of incurring this error.
Traceback (most recent call last):
  File "train.py", line 114, in <module>
    main(argments)
  File "train.py", line 96, in main
    callbacks=get_call_back('rcnn'))
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/training.py", line 1418, in fit_generator
    initial_epoch=initial_epoch)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/training_generator.py", line 217, in fit_generator
    class_weight=class_weight)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/engine/training.py", line 1217, in train_on_batch
    outputs = self.train_function(ins)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 2715, in __call__
    return self._call(inputs)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 2675, in _call
    fetched = self._callable_fn(*array_vals)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1454, in __call__
    self._session._session, self._handle, args, status, None)
  File "/root/anaconda3/envs/keras/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py", line 519, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.InvalidArgumentError: TensorArray rcnn_target/map/TensorArray_5_175757@training/SGD/gradients: Could not read from TensorArray index 1.  Furthermore, the element shape is not fully defined: <unknown>.  It is possible you are working with a resizeable TensorArray and stop_gradients is not allowing the gradients to be written.  If you set the full element_shape property on the forward TensorArray, the proper all-zeros tensor will be returned instead of incurring this error.
	 [[Node: training/SGD/gradients/rcnn_target/map/while/TensorArrayWrite_2/TensorArrayWriteV3_grad/TensorArrayReadV3 = TensorArrayReadV3[_class=["loc:@rcnn_target/map/while/TensorArrayWrite_2/TensorArrayWriteV3"], dtype=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"](training/SGD/gradients/rcnn_target/map/while/TensorArrayWrite_2/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3, training/SGD/gradients/rcnn_target/map/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2, training/SGD/gradients/rcnn_target/map/while/Merge_4_grad/Switch:1)]]
```

参考：https://github.com/tensorflow/tensorflow/issues/22448