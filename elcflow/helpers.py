"""主要是帮助做序列化的函数,包含了对dataframe、numpy等一些格式的序列化
"""
import json
import pandas as pd
import numpy as np
from elcflow.defs import *


class ELCJSONEncoder(json.JSONEncoder):
    """主要的序列化器,主要功能就是把一些类型拆成__elc_type__和__elc_data__,后者
    用来记录数据,前者用来恢复数据的时候使用
    """
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return {
                '__elc_type__': ELC_KEY_DATA_TYPE_DATAFRAME,
                '__elc_data__': obj.to_dict()
            }

        if isinstance(obj, np.ndarray):
            return {
                '__elc_type__': ELC_KEY_DATA_TYPE_NDARRAY,
                '__elc_data__': obj.tolist()
            }

        return super(ELCJSONEncoder, self).default(obj)


def json_stringify(dict_obj):
    return json.dumps(dict_obj, cls=ELCJSONEncoder)


def json_parse(dict_obj):
    """TODO: 请注意: 暂时不支持如下的格式: {'x': {'__elc_type__': '', '__elc_data__': {'__elc_type__': '','__elc_data__': ''}}}
    :param dict_obj: 一个plain的dict, 我们需要递归的把各个类型转回来
    :type dict_obj: dict
    :return:
    """
    if isinstance(dict_obj, str):
        dict_obj = json.loads(dict_obj)

    for k, v in dict_obj.items():
        if isinstance(v, dict):
            if '__elc_type__' in v and '__elc_data__' in v:
                # elc 的特殊结构
                _type = v['__elc_type__']
                _data = v['__elc_data__']

                if _type == ELC_KEY_DATA_TYPE_DATAFRAME:
                    dict_obj[k] = pd.DataFrame.from_dict(_data)
                    continue

                if _type == ELC_KEY_DATA_TYPE_NDARRAY:
                    dict_obj[k] = np.array(_data)
                    continue

                raise TypeError('Unknown __elc_type__: {}'.format(_type))
            else:
                json_parse(v)
        else:
            continue
    return dict_obj
