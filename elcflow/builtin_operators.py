"""内置了一些算子,可以当作example
"""
from elcflow.base import register_elc_function


@register_elc_function(outputs=['sum_result'], name='elc_add')
def elc_add(a, b, *args, **kwargs):
    return a + b


@register_elc_function(outputs=['mul_result'], name='elc_mul')
def elc_mul(a, b):
    return a * b


@register_elc_function(outputs=['pow_result'], name='elc_pow')
def elc_pow(x, a=2):
    a = int(a)  # 确保类型正确
    return x ** a


@register_elc_function(outputs=['x'], name='elc_output')
def elc_output(**kwargs):
    print(kwargs)
    return None
