"""实现了Graph的逻辑:包含运行、暂停、snapshot等功能
"""
import copy
import json
from collections import Iterable
from io import BytesIO
import gc

import networkx as nx

from elcflow.base import ELCDataPlaceholder, ELCOperator, ELCNode, ELCEdge
from elcflow.defs import *
from elcflow.builtin_operators import *
from elcflow.helpers import json_stringify, json_parse
from elcflow.utils import LoggingMixin


class ELCDict(object):
    """
    定义ELC使用的字典, 每次setitem的时候会记录变换, 然后调用merge_diff可以把一段变化持久化
    注意我们并没有深度匹配diff算法
    """

    def __init__(self, merge_every_diff=False, track_diff=True, init_dict=None, **kwargs):
        # 也许order和dict可以并在一起
        self.dict_ = {} if init_dict is None else copy.deepcopy(init_dict)
        self.init_dict = {}
        self.diff_dict = {}
        self.diff_order = []
        self.diff_cache = []
        self.default_key = 0
        self.merge_every_step = merge_every_diff
        self.track_diff = track_diff

    def force_update(self, dict_obj):
        """
        强制更新 不记录修改 可能初始化的时候有用
        :param dict_obj:
        :return:
        """
        if isinstance(dict_obj, dict):
            # 其余情况就是None
            return self.dict_.update(dict_obj)

        assert dict_obj is None, "Unsupported type to update dict: {}".format(type(dict_obj))

    def update(self, dict_obj):
        """
        更新我们当成多个步骤的setitem
        :param dict_obj:
        :type dict_obj: dict
        :return:
        """
        for k, v in dict_obj.items():
            self.__setitem__(k, v)

    def __setitem__(self, key, value):
        if self.track_diff:
            # 如果要track diff
            self.diff_cache.append({key: value})
        self.dict_[key] = value
        if self.merge_every_step:
            # 如果每一步都cache
            self.merge_diff()

    def __getitem__(self, item):
        return self.dict_[item]

    def merge_diff(self, key=None):
        """
        用来合并diff 用处是一个operator可能用到了好多次的setitem
        但是我们只关心这个operator合并的状态变换

        :param key:
        :return:
        """
        if key is None:
            key = self.default_key
            self.default_key += 1
        # reduce state
        __merged_diff = {}
        for diff in self.diff_cache:
            __merged_diff.update(diff)

        if key in self.diff_dict:
            raise KeyError("Duplicate key for diff is not allowed!".format(key))

        self.diff_dict[key] = __merged_diff
        # 记录一个顺序
        self.diff_order.append(key)
        # 重置cache
        self.diff_cache = []

    def replay(self, init_state=None, pause_at_key=None, pause_at_order=None):
        """
        重播diff:
        :param init_state:
        :param pause_at_key: 暂停的key
        :param pause_at_order: 第几个需要暂停
        :return: 重播的state的结果
        """
        if init_state is None:
            init_state = self.init_dict
        state = copy.deepcopy(init_state)  # type: dict
        for i, key in enumerate(self.diff_order):
            if pause_at_order == i or pause_at_key == key:
                return state
            state.update(self.diff_dict[key])
        return state

    def __repr__(self):
        return self.dict_.__repr__()

    def __str__(self):
        return self.dict_.__str__()

    def to_dict(self, keep_diff=True):
        if not keep_diff:
            return self.dict_
        return {
            ELC_DICT_DICT: self.dict_,
            ELC_DICT_DIFF_ORDER: self.diff_order,
            ELC_DICT_DIFF_CACHE: self.diff_cache,
            ELC_DICT_DIFF_DICT: self.diff_dict,
            ELC_DICT_DEFAULT_KEY: self.default_key,
            ELC_DICT_MERGE_EVERY_STEP: self.merge_every_step,
            ELC_DICT_TRACK_DIFF: self.track_diff,
            ELC_DICT_INIT_DICT: self.init_dict
        }

    @classmethod
    def load_from_dict(cls, dict_obj):
        """

        :param dict_obj:
        :type dict_obj: dict
        :return: 一个字典
        """
        instance = cls()

        if ELC_DICT_DICT in dict_obj.keys():
            # 整个状态的load
            for k, v in dict_obj.items():
                setattr(instance, k.replace(ELC_DICT_PREFIX, ''), v)
        else:
            instance.dict_ = dict_obj
        return instance


class ELCState:
    """
    定义ELCFlow的状态
    """

    def __init__(self):
        self._globals = ELCDict()
        self._outputs = ELCDict()

    def to_dict(self):
        return {
            '_globals': self._globals.to_dict(),
            '_outputs': self._outputs.to_dict(),
        }

    def get_globals(self):
        return self._globals

    def get_outputs(self):
        return self._outputs

    def set_globals(self, dict_obj):
        self._globals.force_update(dict_obj)

    def set_outputs(self, dict_obj):
        self._outputs.force_update(dict_obj)

    @classmethod
    def load_from_dict(cls, _globals, _outputs):
        instance = cls()
        instance._globals = ELCDict.load_from_dict(_globals)
        instance._outputs = ELCDict.load_from_dict(_outputs)
        gc.collect()
        return instance

    def __repr__(self):
        return str({'_globals': self._globals.__repr__(), '_outputs': self._outputs.__repr__()})

    def __str__(self):
        return str({'_globals': self._globals.__str__(), '_outputs': self._outputs.__str__()})


class ELCGraph(LoggingMixin):
    """
    elcflow使用的图结构

    :ivar state: 整个计算的state: key是node的id value是这个node的输出
            'globals': {},  # 次字段用来存放globals的变量
            'outputs': {},  # 用来存放各个node的output
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
        self.elc_graph_version = kwargs.get(ELC_GRAPH_VERSION, ELC_GRAPH_VERSION_V1)

        self.state = ELCState()

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
            'state': self.state.to_dict(),
            'node_dict': dict([(k, v.to_dict()) for k, v in self.node_dict.items()]),
            'edge_dict': dict([(k, v.to_dict()) for k, v in self.edge_dict.items()]),
            'execution_node_orders': self.execution_node_orders,
            'ip': self.ip,
            'elc_json': self.elc_json
        }

    def set_state(self, _globals=None, _outputs=None):
        # 把当前状态
        self.state.set_globals(_globals)
        self.state.set_outputs(_outputs)

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
    def load_from_dict(cls, model_dict, model=None):
        """
        将json转化成graph: 支持两种模式: 一种原始的纯图的数据(UI界面生成) 另外一种是to_dict所生成的(保留状态)

        :param model_dict:
        :type model_dict: dict
        :param model:
        :type model: ELCGraph
        :return:
        """
        # model为空且有graph可以重建
        need_creation = 'elc_json' in model_dict and model is None

        if need_creation:
            model = cls.create_from_elc_json(model_dict['elc_json'])

        for k, v in model_dict.items():
            if need_creation and k == 'graph':
                # 不需要 graph 已经重建了
                continue

            if k == 'node_dict':
                # 事实上啥都不需要做
                continue

            if k == 'edge_dict':
                # 事实上只有data会不一样
                for _k, _v in v.items():
                    model.edge_dict[_k].update(**_v)
                continue

            if k == 'state':
                # 状态的重构
                model.state = ELCState.load_from_dict(**v)
                gc.collect()
                continue

            setattr(model, k, v)
        return model

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
        self.state.get_outputs().update(data_dict)

    def execute(self, stop_node_id=None):
        """
        执行图的函数:
        :param stop_node_id: 在哪个node暂停
        :type stop_node_id: str
        :return:
        """
        total_executions = len(self.execution_node_orders)

        self.log.debug('Start from ip={}'.format(self.ip))

        while self.ip < total_executions:
            _node_id = self.execution_node_orders[self.ip]

            self.log.debug('About to execute node=[{}]'.format(_node_id))

            if stop_node_id is not None and stop_node_id == _node_id:
                # 在这边暂停
                return

            _node = self.node_dict[_node_id]  # type: ELCDataPlaceholder or ELCOperator
            if isinstance(_node, ELCDataPlaceholder):
                # 应该在 feed_data_dict就给了
                self.ip += 1
                continue

            # operators需要的东东
            _inputs_list = []
            _inputs_dict = {}

            for [u, v] in self.graph.in_edges(_node_id):
                # 从当前的state取出各种输入
                _edge_id = self.graph.edges[u, v]['id']
                _edge = self.edge_dict[_edge_id]  # type: ELCEdge

                self.log.debug('source_output_id={} and target_input_id={}'.format(
                    _edge.source_output_id, _edge.target_input_id))

                try:
                    if _edge.source_output_id is None or _edge.source_output_id.replace(' ', '') == '':
                        # 不需要再取一步 现在这个情况基本也不需要
                        _data = self.state.get_outputs()[_edge.source_id]
                    else:
                        _data = self.state.get_outputs()[_edge.source_id][_edge.source_output_id]
                except KeyError as e:
                    self.log.error('[source_id={}] and [source_output_id={}] does not not exist in state!'.format(
                        _edge.source_id, _edge.source_output_id))
                    raise e

                if _edge.target_input_id:
                    # TODO: 现在的架构下不会有这个分支了
                    _inputs_dict.update({
                        _edge.target_input_id: _data
                    })
                else:
                    if not isinstance(_data, dict):
                        raise TypeError('Return of operator must be a dict --- FXZhang')
                    _inputs_dict.update(_data)
                    # TODO: 以下代码为位置参数考虑
                    # if not isinstance(_data, dict):
                    #     _inputs_dict.update(_data)
                    #     continue
                    # _inputs_list.append(_data)

                if self._cache:
                    # 把数据写到边上
                    _edge.data.set_data(_data)

            if self.elc_graph_version == ELC_GRAPH_VERSION_V1:
                _inputs_dict.update(_node.parameters)
                self.log.debug("_inputs_list={} and _inputs_dict={}".format(_inputs_list, _inputs_dict))
                _outputs = _node.fn(*_inputs_list, **_inputs_dict)

            elif self.elc_graph_version == ELC_GRAPH_VERSION_V2:
                self.log.debug(
                    "_inputs_list={} and global_states={} and result={} and parameters={}".format(
                        _inputs_list, self.state.get_globals(), _inputs_dict, _node.parameters
                    ))
                _outputs = _node.fn(
                    *_inputs_list,  # 暂时为空的 所以不起作用
                    global_states=self.state.get_globals(),  # 所有的globals在次
                    result=_inputs_dict,  # 上一个的结果
                    parameters=_node.parameters  # node的parameters
                )
            else:
                raise ValueError("Unknown version: {}".format(self.elc_graph_version))

            # 我们把globals的变换记录下来
            # globals的变换被算在当前执行的这个node上
            self.state.get_globals().merge_diff(key=_node_id)

            if isinstance(_outputs, dict):
                # V1的情况目前这个情况只会走这个分支
                self.log.debug("set output for {} with {}".format(_node_id, _outputs))
                self.state.get_outputs().update({_node_id: _outputs})
            else:
                if not isinstance(_outputs, tuple):
                    # TODO: 以下代码对于mapping有用
                    # 我们规定只能返回tuple, 否则无法区分到底[1, 2, 3]是3个整数类型的结果还是1个list的结果
                    _outputs = [_outputs]
                # 确保长度是一样的
                assert len(_node.fn.outputs) == len(_outputs)
                # 把这个node的输出结果存到state里(按照实现定好的名称)
                self.log.debug("set output for {} with {}".format(
                    _node_id, dict([(_node.fn.outputs[i], _outputs[i]) for i in range(len(_outputs))])
                ))
                self.state.get_outputs().update({
                    _node_id: dict([(_node.fn.outputs[i], _outputs[i]) for i in range(len(_outputs))])
                })

            # 下一条指令
            self.ip += 1

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

    def plot(self, show=False, with_state=False, test_mode=False):
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
            self.log.debug('node label:', _node.label)
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
                self.log.debug('label :', _label)
                edge = pydot.Edge(src=src_id, dst=dst_id, label=_label)
            else:
                edge = pydot.Edge(src=src_id, dst=dst_id)
            g.add_edge(edge)

        if show:
            png = g.create_png(prog=['dot', '-Gsize=9,9', '-Gdpi=350'])
            sio = BytesIO(png)
            img = mpimg.imread(sio)
            plt.imshow(img)
            if not test_mode:
                plt.axis('off')
                plt.tight_layout()
                plt.show()

        return g
