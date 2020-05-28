[TOC]

## docker环境


```shell
docker pull splendor90/detectron2

```


```
docker run -it --runtime=nvidia \
-v /sdb/tmp:/sdb/tmp --network=host \
--shm-size="1g" --ipc=host \
--name=yizt_abc splendor90/detectron2 /bin/bash

```



## mac 下测试

安装
```shell
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install -e .

```

依赖安装
```shell
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install
```

demo 测试

```shell
export KMP_DUPLICATE_LIB_OK=TRUE
python demo/demo.py --config-file configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml \
  --input /Users/yizuotian/pyspace/notebook/data/detection/009573.jpg \
  --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
```

```shell
export KMP_DUPLICATE_LIB_OK=TRUE
python demo/demo.py --config-file configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml \
  --input /Users/yizuotian/pyspace/notebook/data/detection/009573.jpg \
  --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_4cafe0.pkl MODEL.DEVICE cpu
```



torch.jit.trace
    报错不支持

```shell
:Could not infer type of list element: Only tensors and (possibly nested) tuples of tensors, lists, or dictsare supported as inputs or outputs of traced functions, but instead got value of type Instances. (toTypeInferredIValue at ../torch/csrc/jit/pybind_utils.h:293)
frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) + 135 (0x12af4e787 in libc10.dylib)
frame #1: torch::jit::toTypeInferredIValue(pybind11::handle) + 540 (0x11d215a4c in libtorch_python.dylib)
frame #2: std::__1::__function::__func<torch::jit::tracer::createGraphByTracing(pybind11::function const&, std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> >, pybind11::function const&, bool, torch::jit::script::Module*)::$_0, std::__1::allocator<torch::jit::tracer::createGraphByTracing(pybind11::function const&, std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> >, pybind11::function const&, bool, torch::jit::script::Module*)::$_0>, std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> > (std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> >)>::operator()(std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> >&&) + 436 (0x11d2c4874 in libtorch_python.dylib)
frame #3: torch::jit::tracer::trace(std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> >, std::__1::function<std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> > (std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> >)> const&, std::__1::function<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > (at::Tensor const&)>, bool, torch::jit::script::Module*) + 1717 (0x120d85e55 in libtorch.dylib)
frame #4: torch::jit::tracer::createGraphByTracing(pybind11::function const&, std::__1::vector<c10::IValue, std::__1::allocator<c10::IValue> >, pybind11::function const&, bool, torch::jit::script::Module*) + 361 (0x11d2c1719 in libtorch_python.dylib)
frame #5: void pybind11::cpp_function::initialize<torch::jit::script::initJitScriptBindings(_object*)::$_13, void, torch::jit::script::Module&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, pybind11::function, pybind11::tuple, pybind11::function, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(torch::jit::script::initJitScriptBindings(_object*)::$_13&&, void (*)(torch::jit::script::Module&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, pybind11::function, pybind11::tuple, pybind11::function, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::'lambda'(pybind11::detail::function_call&)::__invoke(pybind11::detail::function_call&) + 319 (0x11d302a1f in libtorch_python.dylib)
frame #6: pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 3324 (0x11cc3678c in libtorch_python.dylib)
<omitting python frames>
frame #27: start + 1 (0x7fff5affa015 in libdyld.dylib)
frame #28: 0x0 + 11 (0xb in ???)

```

## ubuntu下测试

### coco训练

```shell
cd /sdb/tmp/users/yizt/detectron2
cd datasets
ln -s /sdb/tmp/open_dataset/COCO coco
```

```shell

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python tools/train_net.py --num-gpus 4 \
	--config-file configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml

```


### voc训练

```shell
cd /sdb/tmp/users/yizt/detectron2/datasets
ln -s /sdb/tmp/open_dataset/VOCdevkit/VOC2007 VOC2007
ln -s /sdb/tmp/open_dataset/VOCdevkit/VOC2012 VOC2012
```



```shell
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python tools/train_net.py --num-gpus 8 \
	--config-file configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml
```



## CenterNet-better

​       CenterNet-better依赖detectron2

```shell
cd CenterNet-better/datasets
ln -s /sdb/tmp/open_dataset/VOCdevkit/VOC2007 VOC2007
ln -s /sdb/tmp/open_dataset/VOCdevkit/VOC2012 VOC2012
```



```shell
cd CenterNet-better/playground/centernet.res18.voc.512size
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
dl_train --num-gpus 6
```

