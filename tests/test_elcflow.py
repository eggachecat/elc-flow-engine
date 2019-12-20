import logging
import os

import math
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from operator import add
from numpy.testing import assert_raises

from elcflow.base import register_elc_function
from elcflow.graph import *
from elcflow.helpers import json_stringify, json_parse


def test_helpers():
    obj = {
        'a': 'a',
        'b': 1,
        'c': [1, 2, 3],
        'd': {
            'x': 1,
            'y': {
                'z': [1, 2, 3]
            }
        },
        'f': pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]}),
        'g': np.array([[1, 2, 3], [4, 5, 6], [7, 8]])
    }
    obj_str = json_stringify(obj)
    obj_dict = json.loads(obj_str)
    print(obj_dict)
    json_parse(obj_dict)

    assert obj_str == json_stringify(json_parse(obj_dict))

    with assert_raises(TypeError):
        json_parse({
            'x': {
                '__elc_type__': '_WTF_',
                '__elc_data__': '0'
            }
        })

    # 确保带时间的
    df = pd.DataFrame(data={'datetime': ['2016/11/12', '2016/11/13'], 'col2': [3, 4]})
    df['datetime'] = pd.to_datetime(df['datetime'])
    datetime_dict = {
        'df': df
    }
    obj_dict = json_parse(json.loads(json_stringify(datetime_dict)))
    pd.testing.assert_frame_equal(df, obj_dict['df'])


def test_elc_functions():
    # 加法
    elc_sum = register_elc_function(name='elc_sum', inputs=['a', 'b'], outputs=['sum_ab'])(add)
    assert elc_sum(1, 2) == 3

    # 多个返回值
    @register_elc_function(name='elc_power', outputs=['o1', 'o2', 'o3'], parameters={'exponent': 3})
    def elc_power(a, exponent=2):
        return [math.pow(a, y) for y in range(0, exponent + 1)]

    np.testing.assert_array_equal(elc_power(5), [1, 5, 25, 125])


_model_dict = {
    "nodes": [
        {"label": "Input-2", "id": "3362b879", "_elc_node_type": 'data'},
        {"label": "Input-1", "id": "0ea5a129", "_elc_node_type": 'data'},
        {"label": "Add", "id": "8ac87236", "_elc_node_type": 'operator', "_elc_function": "elc_add"},
        {"label": "MUL", "id": "0d1af6ff", "_elc_node_type": 'operator', "_elc_function": "elc_mul"},
        {"label": "POW", "id": "9d1af6ff", "_elc_node_type": 'operator', "_elc_function": "elc_pow", "_elc_parameters": {"a": 4}},
        {"label": "OUTPUT", "id": "1d1af6ff", "_elc_node_type": 'operator', "_elc_function": "elc_output", },
        {"label": "OUTPUT", "id": "2d1af6ff", "_elc_node_type": 'operator', "_elc_function": "elc_output"},
    ],
    "edges": [
        {"source": "0ea5a129", "target": "8ac87236", "id": "74bc97ca", "_elc_source_output_id": '', "_elc_target_input_id": "a"},
        {"source": "3362b879", "target": "8ac87236", "id": "d3645364", "_elc_source_output_id": '', "_elc_target_input_id": "b"},
        {"source": "8ac87236", "target": "0d1af6ff", "id": "b0eb9a9b", "_elc_source_output_id": 'sum_result', "_elc_target_input_id": "a"},
        {"source": "3362b879", "target": "0d1af6ff", "id": "0e6c0fde", "_elc_source_output_id": '', "_elc_target_input_id": "b"},
        {"source": "0d1af6ff", "target": "9d1af6ff", "id": "7e6c0fde", "_elc_source_output_id": 'mul_result', "_elc_target_input_id": "x"},
        {"source": "0d1af6ff", "target": "1d1af6ff", "id": "1e6c0fde", "_elc_target_input_id": "kwargs"},
        {"source": "9d1af6ff", "target": "2d1af6ff", "id": "2e6c0fde", "_elc_target_input_id": "kwargs"},
    ]
}


def test_elc_graph():
    _graph = ELCGraph(debug=True)
    _graph = ELCGraph.create_from_elc_json(_model_dict)
    _graph.compile()
    _graph.feed_data_dict({
        '3362b879': 5,
        '0ea5a129': 6
    })
    _graph.execute()
    _graph.plot(show=False, with_state=True)
    _graph.plot(show=False, with_state=False)
    assert _graph.state.get_outputs()['9d1af6ff']['pow_result'] == 9150625


def test_elc_graph_io():
    _graph = ELCGraph.create_from_elc_json(_model_dict)
    _graph.compile()
    _graph.feed_data_dict({
        '3362b879': 5,
        '0ea5a129': 6
    })
    _graph.execute(stop_node_id='0d1af6ff')
    _graph_dict = _graph.to_dict()
    obj_str = json_stringify(_graph_dict)
    print(obj_str)
    __graph = ELCGraph.load_from_dict(json_parse(json.loads(obj_str)))
    __graph.compile()
    print(__graph.state)
    __graph.execute()
    __graph.plot(show=False, with_state=False)
    __graph.plot(show=False, with_state=True)

    assert __graph.state.get_outputs()['9d1af6ff']['pow_result'] == 9150625


def test_elc_dict():
    dict_ = ELCDict()
    dict_['x'] = 1
    dict_['x'] = 'x'
    dict_['y'] = np.array([1, 2, 3])
    dict_['z'] = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})

    assert dict_['x'] == 'x'
    np.testing.assert_array_equal(dict_['y'], np.array([1, 2, 3]))
    pd.testing.assert_frame_equal(
        dict_['z'], pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    )

    # diff的cache
    assert dict_.diff_cache[0] == {'x': 1}
    assert dict_.diff_cache[1] == {'x': 'x'}
    np.testing.assert_array_equal(dict_.diff_cache[2]['y'], np.array([1, 2, 3]))
    pd.testing.assert_frame_equal(
        dict_.diff_cache[3]['z'], pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    )

    # 做一个merge
    dict_.merge_diff('merge_key_1')
    assert len(dict_.diff_cache) == 0
    assert dict_.diff_dict['merge_key_1']['x'] == 'x'
    np.testing.assert_array_equal(dict_.diff_dict['merge_key_1']['y'], np.array([1, 2, 3]))
    pd.testing.assert_frame_equal(
        dict_.diff_dict['merge_key_1']['z'], pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    )

    dict_['x'] = 'hei-hei'
    dict_.merge_diff('merge_key_2')
    assert len(dict_.diff_cache) == 0
    assert len(dict_.diff_dict.keys()) == 2
    assert dict_.diff_dict['merge_key_2']['x'] == 'hei-hei'

    dict_replay_1 = dict_.replay(pause_at_key='merge_key_2')
    assert dict_replay_1['x'] == 'x'
    np.testing.assert_array_equal(dict_replay_1['y'], np.array([1, 2, 3]))
    pd.testing.assert_frame_equal(
        dict_replay_1['z'], pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    )

    dict_replay_all = dict_.replay(init_state={'x': 2333})
    assert dict_replay_all['x'] == 'hei-hei'
    np.testing.assert_array_equal(dict_replay_all['y'], np.array([1, 2, 3]))
    pd.testing.assert_frame_equal(
        dict_replay_all['z'], pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    )

    # 测试序列化
    dict_str_1 = json_stringify(dict_.to_dict())
    assert dict_str_1 == json_stringify(dict_.load_from_dict(json_parse(dict_str_1)).to_dict())
    dict_from_str = dict_.load_from_dict(json_parse(dict_str_1))
    assert dict_from_str['x'] == 'hei-hei'
    np.testing.assert_array_equal(dict_from_str['y'], np.array([1, 2, 3]))

    pd.testing.assert_frame_equal(
        dict_from_str['z'], pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    )

    # 默认的key是否可以工作
    dict_2_ = ELCDict()
    dict_2_['x'] = 1
    dict_2_.merge_diff()
    assert dict_2_.diff_dict[0] == {'x': 1}

    dict_2_['x'] = 2
    with assert_raises(KeyError):
        dict_2_.merge_diff(0)

    # merge_every_step是否工作
    dict_3_ = ELCDict(merge_every_diff=True)
    dict_3_['x'] = 1
    dict_3_['x'] = 2
    dict_3_['x'] = 3
    assert len(dict_3_.diff_dict.keys()) == 3
    assert dict_3_.diff_dict[1] == {'x': 2}

    dict_4_ = ELCDict.load_from_dict({'x': 1})
    assert dict_4_['x'] == 1


_model_2_globals = {
    "global_variable_1": 5,
    "multiplier": 3
}

_model_dict_v2 = {
    "nodes": [
        {"label": "data-selector", "id": "0ea5a129", "_elc_node_type": 'operator', "_elc_function": 'elc_select_data_v2', "_elc_parameters": {"key": "global_variable_1"}},
        {"label": "add_plus_plus", "id": "8ac87236", "_elc_node_type": 'operator', "_elc_function": "elc_add_plus_plus_v2"},
        {"label": "multiplier x", "id": "0d1af6ff", "_elc_node_type": 'operator', "_elc_function": "elc_mul_v2"},
        {"label": "pow_for_mul", "id": "9d1af6ff", "_elc_node_type": 'operator', "_elc_function": "elc_pow_for_mul_v2", "_elc_parameters": {"a": 4}},
    ],
    "edges": [
        {"source": "0ea5a129", "target": "8ac87236", "id": "74bc97ca"},
        {"source": "8ac87236", "target": "0d1af6ff", "id": "d3645364"},
        {"source": "0d1af6ff", "target": "9d1af6ff", "id": "b0eb9a9b"},
    ]
}


def test_graph_v2():
    _graph = ELCGraph.create_from_elc_json(_model_dict_v2, elc_graph_version=ELC_GRAPH_VERSION_V2)
    assert _graph.elc_graph_version == ELC_GRAPH_VERSION_V2

    _graph.set_state(_globals=_model_2_globals)
    _graph.compile()
    _graph.execute()
    _graph.plot(show=True, with_state=True, test_mode=True)
    # _graph.plot(show=True, with_state=False)
    # assert _graph.state.get_outputs()['9d1af6ff']['pow_result'] == 9150625


# def test_graph_exceptions():
#     # 类型不对应该好搓
#     with assert_raises(TypeError):
#         pass


if __name__ == '__main__':
    test_graph_v2()
    exit()
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(asctime)s %(name)-12s #%(lineno)d@%(funcName)s() %(levelname)-8s %(message)s',
    #     datefmt='%m-%d %H:%M',
    # )
    test_helpers()
    # test_elc_functions()
    # test_elc_graph()
    # test_elc_graph_io()
    test_elc_dict()

    df = pd.DataFrame(data={'col1': [1.0, 2.0], 'col2': [3, 4]})
    print(df.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict())
    print(df.index.dtype)
    # print(df.dtypes)
    exit()
    dtype_dict = df.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()
    print(pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]}).astype(dtype_dict))
