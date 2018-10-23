



### tf.where vs np.where

np.where 返回的是tuple,tuple中的每个元素是list; tuple中list个数就是坐标的维数

```python
print("np.where([True,False,True]):\n{}".format(np.where([True,False,True])))
print("np.where([[True, False,  True],[False, True,  True]]):\n{}".
      format(np.where([[True, False,  True],[False, True,  True]])))
```

结果如下：

```
np.where([True,False,True]):
(array([0, 2], dtype=int64),)
np.where([[True, False,  True],[False, True,  True]]):
(array([0, 0, 1, 1], dtype=int64), array([0, 2, 1, 2], dtype=int64))
```



tf.where返回的是二维tensor,其中第一个维度的长度是坐标的个数，第二维长度是坐标的维数

```python
print("tf.where([True, False,  True]):\n{}".format(sess.run(tf.where([True, False,  True]))))
print("tf.where([[True, False,  True],[False, True,  True]]):\n{}".
      format(sess.run(tf.where([[True, False,  True],[False, True,  True]]))))
```

结果如下：

```
tf.where([True, False,  True]):
[[0]
 [2]]
tf.where([[True, False,  True],[False, True,  True]]):
[[0 0]
 [0 2]
 [1 1]
 [1 2]]
```



