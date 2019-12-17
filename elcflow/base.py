"""各个主要基础元件的定义,包含了函数/算子/数据/节点和边
"""
import inspect
import json
from collections import Iterable
from typing import Callable
import pandas as pd
import numpy as np
from elcflow.defs import *

# 所有的注册的函数都在这
ELC_FUNCTION_DICT = {}


class ELCFunction(object):
    """
        算子使用的函数
    """

    def __init__(self, **kwargs):
        """

        :param fn: 一个可以呼叫的函数
        :param name: 函数使用的名称
        :param inputs: 输入对应的名称 暂时不支持*args, **kwargs
        :param outputs: 输出对应的名称
        :param parameters: 参数的名称(例如一个指数函数pow, pow(x, a=2)中的2就在这个字段)

        """
        self.fn = kwargs.get(ELC_KEY_FUNCTION_FN)  # type: Callable
        self.name = kwargs.get(ELC_KEY_FUNCTION_NAME)  # type: str
        self.inputs = kwargs.get(ELC_KEY_FUNCTION_INPUTS)  # type: list
        self.outputs = kwargs.get(ELC_KEY_FUNCTION_OUTPUTS)  # type: list
        if kwargs.get(ELC_KEY_FUNCTION_PARAMETERS, {}) is None:
            self.parameters = {}
        else:
            self.parameters = kwargs.get(ELC_KEY_FUNCTION_PARAMETERS, {})  # type: dict

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **self.parameters, **kwargs)

    def __repr__(self):
        return "{}(name={}, inputs={}, outputs={}, parameters={})".format(
            self.__class__.__name__, self.name, self.inputs, self.outputs, self.parameters)


def register_elc_function(outputs, inputs=None, name=None, parameters=None):
    assert isinstance(outputs, Iterable), 'outputs must be iterable'

    def _d(fn):
        _inputs = inputs if inputs is not None else list([p[1].name for p in inspect.signature(fn).parameters.items()])
        _f_name = name if name is not None else fn.__name__
        ELC_FUNCTION_DICT[_f_name] = ELCFunction(fn=fn, name=_f_name, outputs=outputs, inputs=_inputs, parameters=parameters)
        return ELC_FUNCTION_DICT[_f_name]

    return _d


class ELCData(object):
    """整个Graph上的数据, 不同数据对应了不同的序列化函数
    """

    def __init__(self, **kwargs):
        self.data = kwargs.get(ELC_KEY_DATA_DATA, None)

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data


class ELCNode(object):
    """节点:
        主要应该规定输入和输出
        输入按照名称来: UI上方便实现
        输出可以说tuple(多个),但是会给他们名称
    """
    __elc_type__ = 'node'

    def __init__(self, **kwargs):
        """
        初始化一个对象, 必须要有一个独一无二的ID(UI界面会自动生成),这样在连接的时候才能保证没有问题

        :param str id: 这个node的id: 例如 52111314

        :param list inputs: 需要的输入的名称

        :param list outputs: 输出的tuple对应的名称

        """

        self.id = kwargs.get(ELC_KEY_NODE_ID)

        # 显示用的
        self.label = kwargs.get(ELC_KEY_NODE_LABEL, self.id)

    def __repr__(self):
        """
        打印可见
        """
        return "ELCNode(id={}, label={})".format(self.id, self.label)

    def to_dict(self):
        return {
            '__elc_type__': self.__elc_type__,
            ELC_KEY_NODE_ID: self.id,
            ELC_KEY_NODE_LABEL: self.label
        }


class ELCDataPlaceholder(ELCNode):
    """存数据的node"""
    __elc_type__ = 'data_place_holder'
    pass


class ELCOperator(ELCNode):
    __elc_type__ = 'operator'

    def __init__(self, **kwargs):
        try:
            self.fn = ELC_FUNCTION_DICT[kwargs.get(ELC_KEY_NODE_FUNCTION)]  # type: ELCFunction
        except KeyError as e:
            raise KeyError('No registered function: {}'.format(kwargs.get(ELC_KEY_NODE_FUNCTION)))
        ELCNode.__init__(self, **kwargs, inputs=self.fn.inputs, outputs=self.fn.outputs)
        self.parameters = kwargs.get(ELC_KEY_NODE_PARAMETER, {})  # type:dict

    def __repr__(self):
        """
        打印可见
        """
        return "{}({}, fn={})".format(self.__class__.__name__, ELCNode.__repr__(self), self.fn.__repr__())

    def to_dict(self):
        return {
            **ELCNode.to_dict(self),
            ELC_KEY_NODE_PARAMETER: self.parameters,
            ELC_KEY_NODE_FUNCTION: self.fn.name
        }


class ELCEdge(object):
    def __init__(self, **kwargs):
        self.id = kwargs.get(ELC_KEY_EDGE_ID)
        self.source_id = kwargs.get(ELC_KEY_EDGE_SOURCE)
        self.target_id = kwargs.get(ELC_KEY_EDGE_TARGET)
        self.name = kwargs.get(ELC_KEY_EDGE_NAME)
        self.source_output_id = kwargs.get(ELC_KEY_EDGE_SOURCE_OUTPUT_ID)
        self.target_input_id = kwargs.get(ELC_KEY_EDGE_TARGET_INPUT_ID)
        self.data = ELCData(data=kwargs.get(ELC_KEY_EDGE_DATA, None))  # type: ELCData

    def to_dict(self):
        return {
            ELC_KEY_EDGE_ID: self.id,
            ELC_KEY_EDGE_SOURCE: self.source_id,
            ELC_KEY_EDGE_TARGET: self.target_id,
            ELC_KEY_EDGE_NAME: self.name,
            ELC_KEY_EDGE_SOURCE_OUTPUT_ID: self.source_output_id,
            ELC_KEY_EDGE_TARGET_INPUT_ID: self.target_input_id,
            ELC_KEY_EDGE_DATA: self.data.get_data()
        }

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'data':
                setattr(self, k, ELCData(**{ELC_KEY_DATA_DATA: v}))
                continue
            setattr(self, k, v)
