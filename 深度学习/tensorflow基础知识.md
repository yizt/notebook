

tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表

tf.get_collection：从一个结合中取出全部变量，是一个列表

tf.add_n：把一个列表的东西都依次加起来

```python
import tensorflow as tf;  
import numpy as np;  
import matplotlib.pyplot as plt;  
 
v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(0))
tf.add_to_collection('loss', v1)
v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('loss', v2)
 
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print tf.get_collection('loss')
	print sess.run(tf.add_n(tf.get_collection('loss')))
```



在有些机器学习程序中我们想要指定某些操作执行的依赖关系，这时我们可以使用`tf.control_dependencies()`来实现。 

```python
with g.control_dependencies([a, b, c]):
  # `d` and `e` will only run after `a`, `b`, and `c` have executed.
  d = ...
  e = ...
```





A `Tensor` that will hold the new value of ‘ref’ after the assignment has completed. 只有当assign()被执行了才会返回新值 下面两个例子看一下就明白了：

```python

import tensorflow as tf
 
def test_1():
    a = tf.Variable([10, 20])
    b = tf.assign(a, [20, 30])
    c = a + [10, 20]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("test_1 run a : ",sess.run(a)) # => [10 20] 
        print("test_1 run c : ",sess.run(c)) # => [10 20]+[10 20] = [20 40] 因为b没有被run所以a还是[10 20]
        print("test_1 run b : ",sess.run(b)) # => ref:a = [20 30] 运行b，对a进行assign
        print("test_1 run a again : ",sess.run(a)) # => [20 30] 因为b被run过了，所以a为[20 30]
        print("test_1 run c again : ",sess.run(c)) # => [20 30] + [10 20] = [30 50] 因为b被run过了，所以a为[20,30], 那么c就是[30 50]

```



关于`tf.GraphKeys.UPDATE_OPS`，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合`tf.control_dependencies`函数使用。主要作用是将不在计算图中的计算增加在计算图中。

```
        weights_update_op = tf.scatter_sub(weights, label, diff_weights)
        with tf.control_dependencies([weights_update_op]):
            weights_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, weights_update_op)
        
        return loss
```



