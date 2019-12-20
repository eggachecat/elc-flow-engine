"""内置了一些算子,可以当作example
"""
from elcflow.base import register_elc_function, register_elc_function_v2


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


@register_elc_function_v2(name='elc_select_data_v2')
def elc_select_data_v2(global_states, result, parameters):
    return {'return': global_states[parameters['key']]}


@register_elc_function_v2(name='elc_add_plus_plus_v2')
def elc_add_plus_plus_v2(global_states, result, parameters):
    return {'return': result['return'] + 1}


@register_elc_function_v2(name='elc_mul_v2')
def elc_mul_v2(global_states, result, parameters):
    _result = result['return'] * int(global_states['multiplier'])
    global_states['elc_mul_v2_result'] = _result
    return {'return': _result}


@register_elc_function_v2(name='elc_pow_for_mul_v2')
def elc_pow_for_mul_v2(global_states, result, parameters):
    _result = global_states['elc_mul_v2_result'] ** int(parameters['a'])
    global_states['elc_pow_for_mul_v2_result'] = _result
    return {'return': _result}


@register_elc_function_v2(name='elc_output_v2')
def elc_output_v2(global_states, result, parameters=None):
    return {'return': None}