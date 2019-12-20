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
    json_parse(obj_dict)

    assert obj_str == json_stringify(json_parse(obj_dict))

    with assert_raises(TypeError):
        json_parse({
            'x': {
                '__elc_type__': '_WTF_',
                '__elc_data__': '0'
            }
        })


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


def test_with_file():
    test_dir = os.path.dirname(__file__)

    with open(os.path.join(test_dir, './test_1_flow.json'), encoding='utf-8') as fp:
        _model_dict = json.load(fp)
    _graph = ELCGraph.create_from_elc_json(_model_dict)
    _graph.compile()
    _graph.feed_data_dict({
        'cae4e6db': np.array([5, 6]),
        'a7ee6782': np.array([1, 3]),
    })
    _graph.execute()
    _graph_dict = _graph.to_dict()
    _graph.plot(show=False, with_state=True)

    obj_str = json_stringify(_graph_dict)
    with open(os.path.join(test_dir, './test_1_flow_output.json'), 'r', encoding='utf-8') as fp:
        assert obj_str == fp.read()


if __name__ == '__main__':
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(asctime)s %(name)-12s #%(lineno)d@%(funcName)s() %(levelname)-8s %(message)s',
    #     datefmt='%m-%d %H:%M',
    # )
    # test_helpers()
    # test_elc_functions()
    # test_elc_graph()
    # test_elc_graph_io()
    test_with_file()
