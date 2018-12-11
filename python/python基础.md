[TOC]



## python环境配置



### pip



```
wget https://www.python.org/ftp/python/2.7.15/Python-2.7.15.tgz
https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
wget https://pypi.python.org/packages/9d/60/e19cf81ad1743cebea5a799d4ff7179b705b949844c841975e3be4bbb26e/setuptools-28.6.0.zip

wget https://pypi.python.org/packages/source/p/pip/pip-18.0.tar.gz
```



```
yum install readline readline-devel readline-static -y
yum install openssl openssl-devel openssl-static -y
yum install sqlite-devel -y
yum install bzip2-devel bzip2-libs -y
yum install gcc-c++ -y
```



```
tar -xvf Python-2.7.15.tgz
cd Python-2.7.15
./configure --prefix=/usr/local/python-2.7.5
sudo make && sudo make install
```





```
unzip setuptools-28.6.0.zip
cd setuptools-28.6.0
python setup.py install
```



```v
tar -xvf pip-18.0.tar.gz
cd pip*
python setup.py build  
python setup.py install 
```





镜像配置, vi ~/.pip/pip.conf,不存在，就创建此文件，内容如下

```wiki
[global]
timeout = 6000
index-url = http://pypi.douban.com/simple/
[install]
trusted-host = pypi.douban.com
```



pip迁移

```
pip freeze > requirements.txt

pip install -r requirements.txt

```



### conda

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```



conda迁移

```
conda env export > environment.yaml

conda env create -f environment.yaml
```





## 基础编程



### 压缩文件读写

### xml读写



### 图像文件IO

```
skimage.io
plt
opencv
pillow
keras.preprocessing.image
```



### 常用函数

#### zip

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

zip 语法：

```
zip([iterable, ...])
```

参数说明：

- iterabl -- 一个或多个迭代器;

返回值: 

返回元组列表。

```
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```



```
l = ['a', 'b', 'c', 'd', 'e','f']
#打印列表
list(zip(l[:-1],l[1:]))
```

结果

```
[('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f')]
```



```
nums = ['flower','flow','flight']
for i in zip(*nums):
    print(i)
```

结果

```
('f', 'f', 'f')
('l', 'l', 'l')
('o', 'o', 'i')
('w', 'w', 'g')
```





## web框架

http://klen.github.io/py-frameworks-bench/



## 综合

### 参数传递

形参*args, **kwargs

*args：（表示的就是将实参中按照位置传值，多出来的值都给args，且以元祖的方式呈现）
**kwargs：（表示的就是形参中按照关键字传值把多余的传值以字典的方式呈现）

```python
def foo(x,*args,**kwargs):
    print(x)
    print(args)
    print(kwargs)
foo(1,2,3,4,y=1,a=2,b=3,c=4)
```

结果

```
1
(2, 3, 4)
{'y': 1, 'a': 2, 'b': 3, 'c': 4}

```



### 导入

相对路径导入

绝对路径导入



### py2to3

30/4