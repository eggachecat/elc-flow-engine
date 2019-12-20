"""用来enum常用的字符串变量
"""
ELC_KEY_NODE_TYPE = '_elc_node_type'
ELC_KEY_NODE_FUNCTION = '_elc_function'
ELC_KEY_NODE_PARAMETER = '_elc_parameters'
ELC_KEY_NODE_LABEL = 'label'
ELC_KEY_NODE_ID = 'id'

ELC_KEY_EDGE_ID = 'id'
ELC_KEY_EDGE_SOURCE = 'source'
ELC_KEY_EDGE_TARGET = 'target'
ELC_KEY_EDGE_NAME = 'name'
ELC_KEY_EDGE_SOURCE_OUTPUT_ID = '_elc_source_output_id'
ELC_KEY_EDGE_TARGET_INPUT_ID = '_elc_target_input_id'
ELC_KEY_EDGE_DATA = 'data'

ELC_KEY_DATA_DATA = 'data'
ELC_KEY_DATA_TYPE = 'type'
ELC_KEY_DATA_TYPE_DATAFRAME = 'dataframe'
ELC_KEY_DATA_TYPE_SERIES = 'series'
ELC_KEY_DATA_TYPE_NDARRAY = 'NDARRAY'

ELC_KEY_FUNCTION_FN = 'fn'
ELC_KEY_FUNCTION_NAME = 'name'
ELC_KEY_FUNCTION_INPUTS = 'inputs'
ELC_KEY_FUNCTION_OUTPUTS = 'outputs'
ELC_KEY_FUNCTION_PARAMETERS = 'parameters'

ELC_GRAPH_VERSION = 'elc_graph_version'
ELC_GRAPH_VERSION_V1 = 'v1'
# 函数的输入规定好: _global, _result, parameters
ELC_GRAPH_VERSION_V2 = 'v2'

ELC_DICT_PREFIX = '__elc_dict__'
ELC_DICT_DICT = '__elc_dict__dict_'
ELC_DICT_INIT_DICT = '__elc_dict__init_dict'
ELC_DICT_DIFF_ORDER = '__elc_dict__diff_order'
ELC_DICT_DIFF_CACHE = '__elc_dict__diff_cache'
ELC_DICT_DIFF_DICT = '__elc_dict__diff_dict'
ELC_DICT_DEFAULT_KEY = '__elc_dict__default_key'
ELC_DICT_MERGE_EVERY_STEP = '__elc_dict__merge_every_step'
ELC_DICT_TRACK_DIFF = '__elc_dict__track_diff'
