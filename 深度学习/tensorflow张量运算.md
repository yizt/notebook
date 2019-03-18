[TOC]



```
tf.reduce_any
tf.split
tf.add_n
tf.stack
tf.concat
tf.gather
tf.gather_nd
tf.nn.top_k
tf.shape
tf.reshape
tf.tile
tf.identity
tf.boolean_mask
tf.image.non_max_suppression
tf.image.crop_and_resize
tf.pad
tf.squeeze
tf.expand_dims
tf.where
tf.stop_gradient
tf.equal
tf.greater
```



### tf.gather、tf.gather_nd

gather将每个坐标的每一维都当做一维坐标去取值

gather_nd直接使用做坐标取值

```
x = tf.Variable([[1,3,1], [0,0,2], [1,3,2]], name='x')
print("x:\n{}".format(sess.run(x)))
print("tf.where(tf.greater(x,3)):\n{}".format(sess.run(tf.where(tf.greater(x,1)))))

```

结果:

```
x:
[[1 3 1]
 [0 0 2]
 [1 3 2]]
 
tf.where(tf.greater(x,3)):
[[0 1]
 [1 2]
 [2 1]
 [2 2]]
```



```
print("tf.gather(x,tf.where(tf.greater(x,3))):\n{}".
      format(sess.run(tf.gather(x,tf.where(tf.greater(x,1))))))
print("tf.gather_nd(x,tf.where(tf.greater(x,3))):\n{}".
      format(sess.run(tf.gather_nd(x,tf.where(tf.greater(x,1))))))
```



```python
tf.gather(x,tf.where(tf.greater(x,3))):
[[[1 3 1]
  [0 0 2]]

 [[0 0 2]
  [1 3 2]]

 [[1 3 2]
  [0 0 2]]

 [[1 3 2]
  [1 3 2]]]
  
 
 tf.gather_nd(x,tf.where(tf.greater(x,3))):
[3 2 3 2]
```



### tf.identity

参考：https://blog.csdn.net/hu_guan_jie/article/details/78495297

 Return a tensor with the same shape and contents as input.  



### tf.reduce_any

   在张量的维度上计算元素的 "逻辑或"。 

```
x = tf.constant([[True,  True], [False, False]])
tf.reduce_any(x)  # True
tf.reduce_any(x, 0)  # [True, True]
tf.reduce_any(x, 1)  # [True, False]

```






参考：https://blog.csdn.net/lenbow/article/details/52181159、https://blog.csdn.net/lenbow/article/details/52152766



### tf.scatter_nd

```
scatter_nd(indices,updates,shape,name=None)
```

根据indices将updates散布到新的（初始为零）张量。

根据索引对给定shape的零张量中的单个值或切片应用稀疏updates来创建新的张量。此运算符是[tf.gather_nd](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-ctv72eru.html)运算符的反函数，它从给定的张量中提取值或切片。

```
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
  print(sess.run(scatter))
```

结果如下：

```
[0, 11, 0, 10, 9, 0, 0, 12]
```



### tf.unique_with_counts函数

```
tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx, count = unique_with_counts(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
count ==> [2, 1, 3, 1, 2]
```

