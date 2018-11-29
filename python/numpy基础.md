[TOC]

meshgrid 应用场景

numpy broadcast使用

np.where嵌套: 

np.where(y>2,2,np.where(y<2,1,y)) 



### 输入输出

1：save、load

2：savez、load    #保存多个numpy变量

3：savetxt、loadtxt   # 文本格式保存

4：tostring、fromstring   # 转字节码

5：tolist   # 转python list

6：array_str   # 转str

7：binary_repr、base_repr   # 二进制表示、任意进制表示





### 数组创建

1:  empty、ones、zeros

2：identity、eye

3：asmatrix、asfarray、asscalar

4：copy、array、asrray

5：linespace、logspace

6：diag、diagonal  #获得对角元素

7：diagflat、diag   #根据对角元素创建对角矩阵

8：tri   # 创建下三角矩阵

9：tril、triu       #将矩阵转为上三角，下三角



### 数组操作

1：reshape、flatten、ravel

2：transpose、swapaxes

3:   expand_dims、squeeze    #  扩维，压维

4：concatenate、vstack、hstack、dstack、stack(会增加维度)

concatenate在已经存在的维度上连接,除了连接的维度，其它维度必须相等

```
Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
```

stack在一个新的维度上连接数组

```
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.stack((a, b))
array([[1, 2, 3],
       [2, 3, 4]])

>>> np.stack((a, b), axis=-1)
array([[1, 2],
       [2, 3],
       [3, 4]])
```

vstack在第一维上连接，返回最少二维，如果输入维度大于2位，不产生新的维度

```
Examples
--------
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])

>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[2], [3], [4]])
>>> np.vstack((a,b))
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])
```

hstack 在列上做连接，不会增加维度;输入维度>=2等价于np.concatenate((a,b),axis=1)

```
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.hstack((a,b))
array([1, 2, 3, 2, 3, 4])
>>> a = np.array([[1],[2],[3]])
>>> b = np.array([[2],[3],[4]])
>>> np.hstack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])
```









5：split、hsplit、vsplit、dsplit

6：tile、repeat

repeat只能在一个维度上进行元素级别的复制，没有指定轴，就打平数组

```
Parameters
----------
a : array_like
    Input array.
repeats : int or array of ints
    The number of repetitions for each element.  `repeats` is broadcasted
    to fit the shape of the given axis.
    repeats在指定轴的形状上广播
axis : int, optional
    The axis along which to repeat values.  By default, use the
    flattened input array, and return a flat output array.
    重复的轴，默认打平输入，打平输出

>>> np.repeat(3, 4)
array([3, 3, 3, 3])
>>> x = np.array([[1,2],[3,4]])
>>> np.repeat(x, 2)
array([1, 1, 2, 2, 3, 3, 4, 4])
>>> np.repeat(x, 3, axis=1)
array([[1, 1, 1, 2, 2, 2],
       [3, 3, 3, 4, 4, 4]])
>>> np.repeat(x, [1, 2], axis=0)
array([[1, 2],
       [3, 4],
       [3, 4]])
>>>x=np.reshape(np.arange(24),(2,3,4))
>>>np.repeat(x,[3,4],axis=0).shape
(7, 3, 4)
```



tile

在维度上复制数组

```
 tile(A, reps)

Construct an array by repeating A the number of times given by reps.

If `reps` has length ``d``, the result will have dimension of
``max(d, A.ndim)``.

If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
or shape (1, 1, 3) for 3-D replication. If this is not the desired
behavior, promote `A` to d-dimensions manually before calling this
function.
提升A维度
If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
(1, 1, 2, 2).
提升reps维度

Examples
--------
>>> a = np.array([0, 1, 2])
>>> np.tile(a, 2)
array([0, 1, 2, 0, 1, 2])
>>> np.tile(a, (2, 2))
array([[0, 1, 2, 0, 1, 2],
       [0, 1, 2, 0, 1, 2]])
>>> np.tile(a, (2, 1, 2))
array([[[0, 1, 2, 0, 1, 2]],
       [[0, 1, 2, 0, 1, 2]]])

>>> b = np.array([[1, 2], [3, 4]])
>>> np.tile(b, 2)
array([[1, 2, 1, 2],
       [3, 4, 3, 4]])
>>> np.tile(b, (2, 1))
array([[1, 2],
       [3, 4],
       [1, 2],
       [3, 4]])

>>> c = np.array([1,2,3,4])
>>> np.tile(c,(4,1))
array([[1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4]])
```





7：flip、fliplr、flipud

8：rot90、roll



### 线性代数

1：dot、matmul、inner、tensordot



matmul

```python
>>> a = np.arange(2*2*4).reshape((2,2,4))
>>> b = np.arange(2*2*4).reshape((2,4,2))
>>> np.matmul(a,b).shape
(2, 2, 2)
```



dot：通用矩阵乘法

```python
>>> a = np.arange(2*2*4).reshape((2,2,4))
>>> b = np.arange(2*2*4).reshape((2,4,2))
>>> np.dot(a,b).shape
(2, 2, 2, 2)
```



tensordot: 张量点积

```python
>>> a = np.arange(60.).reshape(3,4,5)
>>> b = np.arange(24.).reshape(4,3,2)
>>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
>>> c.shape
(5, 2)
>>> c
array([[ 4400.,  4730.],
       [ 4532.,  4874.],
       [ 4664.,  5018.],
       [ 4796.,  5162.],
       [ 4928.,  5306.]])
>>> # A slower but equivalent way of computing the same...
>>> d = np.zeros((5,2))
>>> for i in range(5):
...   for j in range(2):
...     for k in range(3):
...       for n in range(4):
...         d[i,j] += a[k,n,i] * b[n,k,j]
>>> c == d
```



inner：向量点积，需要最后一维相同

```python
Parameters
----------
a, b : array_like
    If `a` and `b` are nonscalar, their last dimensions must match.

Returns
-------
out : ndarray
    `out.shape = a.shape[:-1] + b.shape[:-1]`

A multidimensional example:

>>> a = np.arange(24).reshape((2,3,4))
>>> b = np.arange(4)
>>> np.inner(a, b)
array([[ 14,  38,  62],
       [ 86, 110, 134]])
```



2: vdot

```python
np.vdot(x,y) 等价于 (x*y).sum()
Note that `vdot` handles multidimensional arrays differently than `dot`:
it does *not* perform a matrix product, but flattens input arguments
to 1-D vectors first. Consequently, it should only be used for vectors.
```



3: outer

```
Compute the outer product of two vectors.

Returns
-------
out : (M, N) ndarray
    ``out[i, j] = a[i] * b[j]``
An example using a "vector" of letters:

>>> x = np.array(['a', 'b', 'c'], dtype=object)
>>> np.outer(x, [1, 2, 3])
array([[a, aa, aaa],
       [b, bb, bbb],
       [c, cc, ccc]], dtype=object)
```



4: cholesky

```python
Cholesky 分解是把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解。它要求矩阵的所有特征值必须大于零，故分解的下三角的对角元也是大于零的。
x = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.int32)
l = np.linalg.cholesky(x)
x1=np.dot(l,l.T.conjugate())
x == x1
```



5: qr

```python
如果实（复）非奇异矩阵A能够化成正交（酉）矩阵Q与实（复）非奇异上三角矩阵R的乘积，即A=QR，则称其为A的QR分解。
x = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32)
q,r = np.linalg.qr(x)
```



6: svd

```python
奇异值分解
x = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 2, 0, 0, 0]], dtype=np.float32)
U,s,V = np.linalg.svd(x)
```



7: eig

```python
# 特征值，特征向量
x = np.diag((1, 2, 3))
eigenvals,eigenvecs = np.linalg.eig(x)
print (np.array_equal(np.dot(x, eigenvecs), eigenvals * eigenvecs))
```



8: matrix_rank  

```python
# 矩阵的秩
x = np.eye(4)
np.linalg.matrix_rank(x)
```



9: trace

```python
# 矩阵的迹
x = np.eye(4)
np.trace(x)
```



10: inv

```python
# 矩阵的逆
x = np.array([[1., 2.], [3., 4.]])
inv =np.linalg.inv(x)
np.allclose(np.dot(x,inv),np.eye(2))
```



11: det

```python
# 矩阵的行列式
x = np.arange(1, 5).reshape((2, 2))
np.linalg.det(x)
```















参考：<a href=https://www.yiibai.com/numpy>NumPy教程</a>

