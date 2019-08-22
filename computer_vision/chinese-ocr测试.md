



```shell
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cudnn-5.1/cuda/lib64:$LD_LIBRARY_PATH

export CUDNN_PATH="/usr/local/cudnn-5.1/cuda/lib64/libcudnn.so.5"
```



```shell
2018-07-10 18:02:29.472988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080
major: 6 minor: 1 memoryClockRate (GHz) 1.835
pciBusID 0000:02:00.0
Total memory: 7.93GiB
Free memory: 5.83GiB
2018-07-10 18:02:29.473019: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2018-07-10 18:02:29.473029: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2018-07-10 18:02:29.473041: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0)
2018-07-10 18:02:31.102309: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0)
2018-07-10 18:02:33.881737: E tensorflow/stream_executor/cuda/cuda_dnn.cc:371] could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
2018-07-10 18:02:33.881800: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:369] driver version file contents: """NVRM version: NVIDIA UNIX x86_64 Kernel Module  390.48  Thu Mar 22 00:42:57 PDT 2018
GCC version:  gcc 版本 4.8.5 20150623 (Red Hat 4.8.5-16) (GCC) 
"""
2018-07-10 18:02:33.881855: I tensorflow/stream_executor/cuda/cuda_dnn.cc:379] possibly insufficient driver version: 390.48.0
2018-07-10 18:02:33.881869: E tensorflow/stream_executor/cuda/cuda_dnn.cc:338] could not destroy cudnn handle: CUDNN_STATUS_BAD_PARAM
2018-07-10 18:02:33.881879: F tensorflow/core/kernels/conv_ops.cc:672] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms) 
[I 18:02:34.687 NotebookApp] KernelRestarter: restarting kernel (1/5), keep random ports

```



2:pytorch_binding安装报错

```
cd ../pytorch_binding
python setup.py install

```





```shell
/root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/generic/THCTensorMasked.h:14:41: 附注：in expansion of macro ‘real’
                                         real value);
                                         ^
In file included from /root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THCGenerateAllTypes.h:22:0,
                 from /root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THCTensorMath.h:38,
                 from /root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THC.h:16,
                 from build/warpctc_pytorch/_warp_ctc/__warp_ctc.c:493:
/root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THCGenerateHalfType.h:9:14: 错误：未知的类型名‘half’
 #define real half
              ^
/root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/generic/THCTensorScatterGather.h:7:108: 附注：in expansion of macro ‘real’
 THC_API void THCTensor_(scatterFill)(THCState* state, THCTensor *tensor, int dim, THCudaLongTensor *index, real value);
                                                                                                            ^
In file included from /root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THCGenerateAllTypes.h:22:0,
                 from /root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THCTensorMath.h:41,
                 from /root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THC.h:16,
                 from build/warpctc_pytorch/_warp_ctc/__warp_ctc.c:493:
/root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THCGenerateHalfType.h:9:14: 错误：未知的类型名‘half’
 #define real half
              ^
/root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/generic/THCTensorIndex.h:7:106: 附注：in expansion of macro ‘real’
 THC_API void THCTensor_(indexFill)(THCState *state, THCTensor *tensor, int dim, THCudaLongTensor *index, real val);
                                                                                                          ^
/root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/THCGenerateHalfType.h:9:14: 错误：未知的类型名‘half’
 #define real half
              ^
/root/anaconda3/envs/chinese-ocr/lib/python2.7/site-packages/torch/utils/ffi/../../lib/include/THC/generic/THCTensorIndex.h:12:107: 附注：in expansion of macro ‘real’
 THC_API void THCTensor_(indexFill_long)(THCState *state, THCTensor *tensor, int dim, THLongTensor *index, real val);
                                                                                                           ^
error: command 'gcc' failed with exit status 1

```



原因：

python版本不对

```
conda install pytorch=0.3.0 cuda80 torchvision -c soumith
```





训练



mkdir train/keras-train/save_model