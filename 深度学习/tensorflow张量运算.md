



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



