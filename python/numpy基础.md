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

5：split、hsplit、vsplit、dsplit

6：tile、repeat

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

