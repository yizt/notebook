





## 错误列表

1: 安装报错

```
./install.sh
```





```shell
[ 12%] Building NVCC (Device) object lib/THC/CMakeFiles/THC.dir/THC_generated_THCTensorMathBlas.cu.o
/root/torch/extra/cutorch/lib/THC/generic/THCTensorMath.cu(393): error: more than one operator "==" matches these operands:
            function "operator==(const __half &, const __half &)"
            function "operator==(half, half)"
            operand types are: half == half

/root/torch/extra/cutorch/lib/THC/generic/THCTensorMath.cu(414): error: more than one operator "==" matches these operands:
            function "operator==(const __half &, const __half &)"
            function "operator==(half, half)"
            operand types are: half == half

2 errors detected in the compilation of "/tmp/tmpxft_00002ada_00000000-6_THCTensorMath.cpp1.ii".
CMake Error at THC_generated_THCTensorMath.cu.o.cmake:267 (message):
  Error generating file
  /root/torch/extra/cutorch/build/lib/THC/CMakeFiles/THC.dir//./THC_generated_THCTensorMath.cu.o


make[2]: *** [lib/THC/CMakeFiles/THC.dir/THC_generated_THCTensorMath.cu.o] 错误 1
make[2]: *** 正在等待未完成的任务....
make[1]: *** [lib/THC/CMakeFiles/THC.dir/all] 错误 2
make: *** [all] 错误 2

Error: Build error: Failed building.

```



解决方法

```shell
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
./install.sh
```



参考：https://www.jianshu.com/p/9ab8c93275b7



2： 测试报错

```shell
./test.sh
```



```shell
211/212 CosineEmbeddingCriterion ........................................ [PASS]
212/212 SpatialFractionalMaxPooling ..................................... [PASS]
Completed 167836 asserts in 212 tests with 0 failures and 0 errors
sundown loaded succesfully
cutorch loaded succesfully
cunn loaded succesfully
/root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/cudnn/ffi.lua:1603: 'libcudnn (R5) not found in library path.
Please install CuDNN from https://developer.nvidia.com/cuDNN
Then make sure files named as libcudnn.so.5 or libcudnn.5.dylib are placed in
your library load path (for example /usr/local/lib , or manually add a path to LD_LIBRARY_PATH)

Alternatively, set the path to libcudnn.so.5 or libcudnn.5.dylib
to the environment variable CUDNN_PATH and rerun torch.
For example: export CUDNN_PATH="/usr/local/cuda/lib64/libcudnn.so.5"

stack traceback:
	[C]: in function 'error'
	/root/torch/install/share/lua/5.1/cudnn/ffi.lua:1603: in main chunk
	[C]: in function 'require'
	/root/torch/install/share/lua/5.1/cudnn/init.lua:4: in main chunk
	[C]: at 0x0046b030
	[C]: at 0x004064f0

```



依赖cudnn版本为5.x，依赖cuda 8.0 或7.5;目前我们安装的是7.x



```shell
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
mv cuda_8.0.61_375.26_linux-run cuda_8.0.61_375.26_linux.run
chmod 755 *
./cuda_8.0.61_375.26_linux.run --kernel-source-path='/usr/src/kernels/3.10.0-693.el7.x86_64'

```



```wiki
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: n

Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
 [ default is /usr/local/cuda-8.0 ]: 

Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: n

Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: n
```



如下方式执行：

```shell
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cudnn-5.1/cuda/lib64:$LD_LIBRARY_PATH

export CUDNN_PATH="/usr/local/cudnn-5.1/cuda/lib64/libcudnn.so.5"

```



3: 测试报错



```shell
./test.sh
```



```shell
PReLU_backward
 Function call failed
/root/torch/install/share/lua/5.1/nn/THNN.lua:110: Wrong number of input planes. Expected 8 but got 10. at /tmp/luarocks_nn-scm-1-6879/nn/lib/THNN/generic/PReLU.c:29
stack traceback:
	[C]: in function 'v'
	/root/torch/install/share/lua/5.1/nn/THNN.lua:110: in function 'PReLU_updateOutput'
	/root/torch/install/share/lua/5.1/nn/PReLU.lua:12: in function 'forward'
	/root/torch/install/share/lua/5.1/cunn/test.lua:5545: in function 'v'
	/root/torch/install/share/lua/5.1/cunn/test.lua:6670: in function </root/torch/install/share/lua/5.1/cunn/test.lua:6668>
	[C]: in function 'xpcall'
	/root/torch/install/share/lua/5.1/torch/Tester.lua:477: in function '_pcall'
	/root/torch/install/share/lua/5.1/torch/Tester.lua:436: in function '_run'
	/root/torch/install/share/lua/5.1/torch/Tester.lua:355: in function 'run'
	/root/torch/install/share/lua/5.1/cunn/test.lua:6691: in function 'testcuda'
	[string "nn.testcuda()"]:1: in main chunk
	[C]: in function 'pcall'
	/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:136: in main chunk
	[C]: at 0x004064f0

--------------------------------------------------------------------------------
PReLU_forward
 Function call failed
/root/torch/install/share/lua/5.1/nn/THNN.lua:110: Wrong number of input planes. Expected 8 but got 46. at /tmp/luarocks_nn-scm-1-6879/nn/lib/THNN/generic/PReLU.c:29
stack traceback:
	[C]: in function 'v'
	/root/torch/install/share/lua/5.1/nn/THNN.lua:110: in function 'PReLU_updateOutput'
	/root/torch/install/share/lua/5.1/nn/PReLU.lua:12: in function 'forward'
	/root/torch/install/share/lua/5.1/cunn/test.lua:5519: in function 'v'
	/root/torch/install/share/lua/5.1/cunn/test.lua:6670: in function </root/torch/install/share/lua/5.1/cunn/test.lua:6668>
	[C]: in function 'xpcall'
	/root/torch/install/share/lua/5.1/torch/Tester.lua:477: in function '_pcall'
	/root/torch/install/share/lua/5.1/torch/Tester.lua:436: in function '_run'
	/root/torch/install/share/lua/5.1/torch/Tester.lua:355: in function 'run'
	/root/torch/install/share/lua/5.1/cunn/test.lua:6691: in function 'testcuda'
	[string "nn.testcuda()"]:1: in main chunk
	[C]: in function 'pcall'
	/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:136: in main chunk
	[C]: at 0x004064f0

--------------------------------------------------------------------------------
/root/torch/install/share/lua/5.1/torch/Tester.lua:361: An error was found while running tests!	

```



尚未解决：

参考：https://github.com/torch/cunn/issues/474

https://github.com/torch/torch7/issues/1051



4：测试报错

```shell
th train.lua -dataset USPS -eta 0.9
```





```shell
/root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/trepl/init.lua:389: module 'hdf5' not found:No LuaRocks module found for hdf5
	no field package.preload['hdf5']
	no file '/root/.luarocks/share/lua/5.1/hdf5.lua'
	no file '/root/.luarocks/share/lua/5.1/hdf5/init.lua'
	no file '/root/torch/install/share/lua/5.1/hdf5.lua'
	no file '/root/torch/install/share/lua/5.1/hdf5/init.lua'
	no file './hdf5.lua'
	no file '/root/torch/install/share/luajit-2.1.0-beta1/hdf5.lua'
	no file '/usr/local/share/lua/5.1/hdf5.lua'
	no file '/usr/local/share/lua/5.1/hdf5/init.lua'
	no file '/root/.luarocks/lib/lua/5.1/hdf5.so'
	no file '/root/torch/install/lib/lua/5.1/hdf5.so'
	no file '/root/torch/install/lib/hdf5.so'
	no file './hdf5.so'
	no file '/usr/local/lib/lua/5.1/hdf5.so'
	no file '/usr/local/lib/lua/5.1/loadall.so'
stack traceback:
	[C]: in function 'error'
	/root/torch/install/share/lua/5.1/trepl/init.lua:389: in function 'require'
	train.lua:7: in main chunk
	[C]: in function 'dofile'
	/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:150: in main chunk
	[C]: at 0x004064f0

```





```
luarocks install hdf5
```



 任然报错

```
[root@localhost JULE.torch]# th train.lua -dataset USPS -eta 0.9
/root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/trepl/init.lua:389: /root/torch/install/share/lua/5.1/hdf5/ffi.lua:71: Unsupported HDF5 version: 1.10.1
stack traceback:
	[C]: in function 'error'
	/root/torch/install/share/lua/5.1/trepl/init.lua:389: in function 'require'
	train.lua:7: in main chunk
	[C]: in function 'dofile'
	/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:150: in main chunk
	[C]: at 0x004064f0

```



解决方法：

```shell
git clone https://github.com/anibali/torch-hdf5.git
cd torch-hdf5
git checkout hdf5-1.10 
luarocks make hdf5-0-0.rockspec
```



参考： https://blog.csdn.net/u013548568/article/details/79732856