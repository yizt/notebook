

<!--
 * @Author: yizuotian
 * @Date: 2024-06-11 10:41:25
 * @LastEditors: yizuotian
 * @LastEditTime: 2024-07-17 13:47:18
 * @FilePath: /notebook/jax_examples/README.md
 * @Description: Do edit!
-->



<!--
 * @Author: yizuotian
 * @Date: 2024-06-11 10:41:25
 * @LastEditors: yizuotian
 * @LastEditTime: 2024-06-11 11:02:17
 * @FilePath: /jax_examples/README.md
 * @Description: Do edit!
-->

[toc]


```shell
conda activate rl310
```

## XLA
XLA（加速线性代数）是一种针对特定领域的线性代数编译器，能够加快 TensorFlow 模型的运行速度，而且可能完全不需要更改源代码。
它可以提高运行速度并改进内存用量。

运行 TensorFlow 程序后，所有操作均由 TensorFlow 执行程序单独执行。每个 TensorFlow 操作都有一个预编译的 GPU 内核实现，可以将执行程序分派给该实现。

XLA 提供了一种运行模型的替代模式：它会将 TensorFlow 图编译成一系列专门为给定模型生成的计算内核。由于这些内核是模型特有的，因此它们可以利用模型专属信息进行优化。以 XLA 在简单的 TensorFlow 计算环境中进行的优化为例：
```python
def model_fn(x, y, z):
  return tf.reduce_sum(x + y * z)
```
如果在不使用 XLA 的情况下运行，图会启动三个内核：分别对应于乘法、加法和减法运算。但是，XLA 可以优化该图，使其启动一次内核就能计算结果。它通过将加法、乘法和减法“融合”到一个 GPU 内核中来实现这一点。此外，这种融合操作不会将由 y*z 和 x+y*z 生成的中间值写出到内存中；而是直接将这些中间计算的结果“流式传输”给用户，同时将它们完全保留在 GPU 寄存器中。<font color='red'>融合是 XLA 采用的最重要的一项优化措施。 内存带宽通常是硬件加速器上最稀缺的资源，因此消除内存操作是提高性能的最佳方法之一。</font>

通过 `tf.function(jit_compile=True)` 进行明确编译

参考：https://www.tensorflow.org/xla?hl=zh-cn

## jax
JAX: High-Performance Array Computing
JAX is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

If you’re looking to train neural networks, use Flax and start with its documentation. Some associated tools are Optax and Orbax. For an end-to-end transformer library built on JAX, see MaxText.

Familiar API
JAX provides a familiar NumPy-style API for ease of adoption by researchers and engineers.

Transformations
JAX includes composable function transformations for compilation, batching, automatic differentiation, and parallelization.

Run Anywhere
The same code executes on multiple backends, including CPU, GPU, & TPU


