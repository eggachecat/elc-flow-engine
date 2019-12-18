"""实现了Graph的逻辑:包含运行、暂停、snapshot等功能
"""
import json
from collections import Iterable
from io import BytesIO

import networkx as nx

from elcflow.base import ELCDataPlaceholder, ELCOperator, ELCNode, ELCEdge
from elcflow.defs import ELC_KEY_NODE_TYPE
from elcflow.builtin_operators import *
from elcflow.helpers import json_stringify


class ELCGraph:
    """
    elcflow使用的图结构

    :ivar state: 整个计算的state: key是node的id value是这个node的输出
    :ivar data_node_id_list: 所有数据节点的node的id
    :ivar operator_node_id_list: 所有算子节点的node的id
    :ivar node_dict: id -> node
    :ivar edge_dict: id -> edge
    :ivar ip: 模仿编译器: instruction pointer
    :ivar execution_node_orders: 执行的顺序
    :ivar elc_json: UI生成的json(包含了在web上的位置等额外信息)
    """

    def __init__(self, **kwargs):

        self.graph = nx.DiGraph()
        self._debug = kwargs.get("debug", False)
        self._cache = kwargs.get("cache", True)

        self.state = {}

        self.data_node_id_list = []
        self.operator_node_id_list = []

        self.node_dict = {}
        self.edge_dict = {}

        self.ip = 0
        self.execution_node_orders = []

        self.elc_json = {}

    @staticmethod
    def convert_json_to_object(json_dict):
        """
        帮助从json中parse出node和edges

        :param json_dict: 描述图的dict
        :type json_dict: dict
        :return:
        """
        node_object_list = []
        edge_object_list = []
        # 节点
        for _node in json_dict['nodes']:
            _node_type = _node[ELC_KEY_NODE_TYPE]
            if _node_type == 'data':
                node_object_list.append(ELCDataPlaceholder(**_node))
                continue

            if _node_type == 'operator':
                node_object_list.append(ELCOperator(**_node))
                continue

            raise TypeError('Unknown _elc_node_type: {}'.format(_node_type))

        for _edge in json_dict['edges']:
            edge_object_list.append(ELCEdge(**_edge))

        return node_object_list, edge_object_list

    def to_dict(self):
        """
        将当前图的状态保存下来

        :return: 一个字典类型
        """
        return {
            'graph': nx.json_graph.adjacency_data(self.graph),
            'data_node_id_list': self.data_node_id_list,
            'operator_node_id_list': self.operator_node_id_list,
            'state': self.state,
            'node_dict': dict([(k, v.to_dict()) for k, v in self.node_dict.items()]),
            'edge_dict': dict([(k, v.to_dict()) for k, v in self.edge_dict.items()]),
            'execution_node_orders': self.execution_node_orders,
            'ip': self.ip,
            'elc_json': self.elc_json
        }

    @classmethod
    def create_from_elc_json(cls, elc_json, **kwargs):
        """
        从ui生成的json生成一个cls
        :param elc_json:
        :param kwargs:
        :return:
        """
        _nodes, _edges = cls.convert_json_to_object(elc_json)
        _graph = cls(debug=True, **kwargs)
        _graph.elc_json = elc_json
        for _n in _nodes:
            _graph.add_node(_n)
        for _e in _edges:
            _graph.add_edge(_e)
        return _graph

    @classmethod
    def load(cls, model_obj, model=None):
        """
        将json转化成graph: 支持两种模式: 一种原始的纯图的数据(UI界面生成) 另外一种是to_dict所生成的(保留状态)

        :param model_obj:
        :type model_obj: dict
        :param model:
        :type model: ELCGraph
        :return:
        """
        # model为空且有graph可以重建
        need_creation = 'elc_json' in model_obj and model is None

        if need_creation:
            model = cls.create_from_elc_json(model_obj['elc_json'])

        for k, v in model_obj.items():
            if need_creation and k == 'graph':
                # 不需要 graph 已经重建了
                continue

            if k == 'node_dict':
                # 事实上啥都不需要做
                continue

            if k == 'edge_dict':
                # 事实上只有data会不一样
                for _k, _v in v.items():
                    print(_v)
                    model.edge_dict[_k].update(**_v)
                continue

            setattr(model, k, v)
        return model

    @staticmethod
    def convert_json_from_elc_to_nx():
        pass

    @staticmethod
    def convert_json_from_nx_to_elc():
        pass

    def compile(self):
        """解析图并且依照拓扑排序生成执行的顺序"""
        self.execution_node_orders = list(nx.dag.topological_sort(self.graph))
        # TODO: 如果支持平行化

    def feed_data_dict(self, data_dict):
        """
        把外部的数据喂入到图上(这一部必须在execute前执行)
        :param data_dict: 输入的字典, key是对应的input类型的node的id, value就是数据
        :type data_dict: dict
        :return:
        """
        assert set(data_dict.keys()) == set(self.data_node_id_list), 'You must feed all data for data placeholders'
        self.state.update(data_dict)

    def execute(self, stop_node_id=None):
        """
        执行图的函数:
        :param stop_node_id: 在哪个node暂停
        :type stop_node_id: str
        :return:
        """
        total_executions = len(self.execution_node_orders)

        if self._debug:
            print('Start from ip={}'.format(self.ip))

        while self.ip < total_executions:
            _node_id = self.execution_node_orders[self.ip]

            if self._debug:
                print('About to execute node=[{}]'.format(_node_id))

            if stop_node_id is not None and stop_node_id == _node_id:
                # 在这边暂停
                return

            _node = self.node_dict[_node_id]  # type: ELCDataPlaceholder or ELCOperator
            if isinstance(_node, ELCDataPlaceholder):
                # 应该在 feed_data_dict就给了
                self.ip += 1
                continue
            _data_dict = {}
            for [u, v] in self.graph.in_edges(_node_id):
                # 从当前的state取出各种输入
                _edge_id = self.graph.edges[u, v]['id']
                _edge = self.edge_dict[_edge_id]  # type: ELCEdge
                if self._debug:
                    print('source_output_id:', _edge.source_output_id)
                    print('target_input_id:', _edge.target_input_id)
                try:
                    if _edge.source_output_id is None or _edge.source_output_id.replace(' ', '') == '':
                        # 不需要再取一步
                        _data = self.state[_edge.source_id]
                    else:
                        _data = self.state[_edge.source_id][_edge.source_output_id]
                except KeyError as e:
                    print('[source_id={}] and [source_output_id={}] does not not exist in state!'.format(
                        _edge.source_id, _edge.source_output_id))
                    raise e
                _data_dict.update({
                    _edge.target_input_id: _data
                })
                if self._cache:
                    # 把数据写到边上
                    _edge.data.set_data(_data)

            _data_dict.update(_node.parameters)
            _outputs = _node.fn(**_data_dict)

            # TODO: 以下mapping的代码也许放在ELCFunction中更好?
            if not isinstance(_outputs, tuple):
                # 我们规定只能返回tuple, 否则无法区分到底[1, 2, 3]是3个整数类型的结果还是1个list的结果
                _outputs = [_outputs]
            # 确保长度是一样的
            assert len(_node.fn.outputs) == len(_outputs)

            # 把这个node的输出结果存到state里(按照实现定好的名称)
            self.state.update({
                _node_id: dict([(_node.fn.outputs[i], _outputs[i]) for i in range(len(_outputs))])
            })

            # 下一条指令
            self.ip += 1

    def set_state(self, state):
        """
        强行更新state(也许在debug的时候有用)
        :param state:
        :return:
        """
        # 如果从错误中回来可能会用到
        self.state.update(state)

    def add_node(self, node):
        """
        新增node到图上
        :param node: 一个node, 可以从convert_json_to_object中得到
        :type node: ELCNode
        :return:
        """

        # 如果是operator
        if isinstance(node, ELCDataPlaceholder):
            # 数据的node
            self.data_node_id_list.append(node.id)

        if isinstance(node, ELCOperator):
            self.operator_node_id_list.append(node.id)

        self.node_dict[node.id] = node
        self.graph.add_node(node.id, **node.to_dict())

    def add_edge(self, edge):
        """
        新增边到图上
        :param edge:
        :type edge: ELCEdge
        :return:
        """
        self.edge_dict[edge.id] = edge
        self.graph.add_edge(edge.source_id, edge.target_id, id=edge.id)

    def plot(self, show=False, with_state=False):
        """

        :param show: 是否要显示
        :type show: bool
        :param with_state: 是否要把数据加在图片上
        :type with_state: bool
        :return: Dot
        """
        import pydot
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        assert self.graph is not None

        g = pydot.Dot(graph_type="digraph")

        # draw nodes
        for node_id in self.graph.nodes():
            _node = self.node_dict[node_id]
            if self._debug:
                print('node label:', _node.label)
            if isinstance(node_id, ELCDataPlaceholder):
                node = pydot.Node(name=node_id, label=_node.label, shape="rect")
            else:
                node = pydot.Node(name=node_id, label=_node.label, shape="circle")
            g.add_node(node)

        # draw edges
        for src_id, dst_id in self.graph.edges():
            if with_state:
                _edge_id = self.graph.edges[src_id, dst_id]['id']
                _edge = self.edge_dict[_edge_id]  # type: ELCEdge
                _label = json_stringify(_edge.data.get_data())
                if self._debug:
                    print('label :', _label)
                edge = pydot.Edge(src=src_id, dst=dst_id, label=_label)
            else:
                edge = pydot.Edge(src=src_id, dst=dst_id)
            g.add_edge(edge)

        if show:
            png = g.create_png()
            sio = BytesIO(png)
            img = mpimg.imread(sio)
            plt.imshow(img, aspect="equal")
            plt.show()

        return g
