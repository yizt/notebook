#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-11 16:51:00
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-11 16:51:01
@ FilePath: /jax_examples/jit_test.py
@ Description: Do edit!
"""
import jax
import jax.numpy as jnp
import time

global_list = []

def log2(x):
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)



def test_time():
    start=time.time()
    x = jnp.arange(1000000)
    selu(x).block_until_ready()
    print(f'test_time ellapse:{time.time()-start}')

def test_jit_time():
    selu_jit = jax.jit(selu)
    # Pre-compile the function before timing...
    start=time.time()
    x = jnp.arange(1000000)
    selu_jit(x).block_until_ready()
    print(f'test_jit_time ellapse:{time.time()-start}')

def test():
	"""
	@ description: 
	@ param {type} 
	@ return: 
	"""
	# One way to see the sequence of primitives behind a function is using jax.make_jaxpr():
	print(jax.make_jaxpr(log2)(3.0))


if __name__ == "__main__":
	#test()
    test_time()
    test_jit_time()/Users/yizuotian/pyspace/octo/examples/flag_test.py
