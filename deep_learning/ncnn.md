## opencv 安装

1. 下载安装

```shell
git clone https://github.com/opencv/opencv.git
cd opencv
git submodule update --init --recursive


mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF ..

# -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \


make -j6
make install

```

2. 问题处理:ippicv无法下载

修改 3rdparty/ippicv/ippicv.cmake

```text
                 "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/"

```

```text
                 "file:///Users/yizuotian/soft/"
```

cmake -DCMAKE_CXX_COMPILER=g++9 -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..


## ncnn 安装
```
cd <ncnn-root-dir>
mkdir -p build
cd build

# cmake option NCNN_VULKAN for enabling vulkan
cmake -DNCNN_VULKAN=OFF ..

make -j6
```

## ncnn android build

```shell

cd ncnn
mkdir -p build-android
cd build-android
cmake -DCMAKE_TOOLCHAIN_FILE=/Users/admin/Library/Android/sdk/ndk-bundle/build/cmake/android.toolchain.cmake \
-DANDROID_ABI="armeabi-v7a" \
-DANDROID_ARM_NEON=ON \
-DANDROID_PLATFORM=android-14 ..


cmake -DCMAKE_TOOLCHAIN_FILE=/Users/admin/Library/Android/sdk/ndk-bundle/build/cmake/android.toolchain.cmake \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_ARM_NEON=ON \
-DANDROID_PLATFORM=android-21 ..

make -j6
make install
```



## pytorch 模型转换

1. python to onnx
```python
from torchvision import models
import torch
import torch.onnx



def main():
    x = torch.rand(1, 3, 224, 224)
    m = models.resnet50(pretrained=False)
    m.load_state_dict(torch.load('/Users/yizuotian/pretrained_model/resnet50-19c8e357.pth'))
    torch_out = torch.onnx._export(m, x, "resnet50.onnx", export_params=True)


if __name__ == '__main__':
    main()


```

2. simplify onnx model

The exported resnet50.onnx model may contains many redundant operators such as Shape, Gather and Unsqueeze that is not supported in ncnn

```shell
source activate pytorch
pip install onnx-simplifier
python -m onnxsim resnet50.onnx resnet50-sim.onnx 
```

3. onnx to ncnn
```shell
export PATH=/Users/yizuotian/pyspace/ncnn/build/tools:/Users/yizuotian/pyspace/ncnn/build/tools/onnx:$PATH
onnx2ncnn resnet50-sim.onnx resnet50.param resnet50.bin
```

