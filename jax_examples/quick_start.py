"""
@ Author: yizuotian
@ Date: 2024-06-11 11:04:34
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-11 11:04:34
@ FilePath: /jax_examples/quick_start.py
@ Description: Do edit!
"""

import jax.numpy as jnp
from jax import grad


def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    # x = jnp.arange(5.0)
    # print(selu(x))
    
    x_small = jnp.arange(3.)
    derivative_fn = grad(sum_logistic)
    print(derivative_fn(x_small))
    
def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))





if __name__ == "__main__":
	test()
