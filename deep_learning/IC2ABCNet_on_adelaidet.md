[TOC]



### ICDAR15 转为ABCNet格式

#### 1. 生成标注数据

a) 找到IC15标准边框的上下两个边, 在上下两边插值两个点；上下各4个点；直接作为bezier控制点; 顺序如下，上边的点从左到右，下边的点从右到左。

```txt
x0, x1, x2, x3, y0, y1, y2, y3,x0_b, x1_b, x2_b, x3_b, y0_b, y1_b, y2_b, y3_b
```

b)  将标注索引化



#### 2.修改Adelaidet工程代码

a) 修改`adet/data/builtin.py` 中`_PREDEFINED_SPLITS_TEXT` 变量，增加ic15的定义

增加如下两行

```python
	"ic15_train": ("ic15/train_images", "ic15/annotations/train_ic15_maxlen100_v2.json"),
    "ic15_test": ("ic15/test_images","ic15/annotations/test_ic15_maxlen100_v2.json"),
```

修改后如下：

```python
_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "ic15_train": ("ic15/train_images", "ic15/annotations/train_ic15_maxlen100_v2.json"),
    "ic15_test": ("ic15/test_images","ic15/annotations/test_ic15_maxlen100_v2.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
}
```



#### 3. 增加配置文件

a) 复制`configs/BAText/CTW1500` 到`configs/BAText/CTW1500` 

b) 





