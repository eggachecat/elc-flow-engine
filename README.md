 # elcflow
 
 [![Build Status](https://travis-ci.com/eggachecat/elc-flow-engine.svg?branch=master)](https://travis-ci.com/eggachecat/elc-flow-engine)
 [![Coverage Status](https://coveralls.io/repos/github/eggachecat/elc-flow-engine/badge.svg)](https://coveralls.io/github/eggachecat/elc-flow-engine)
 [![Documentation Status](https://readthedocs.org/projects/elc-flow-engine/badge/?version=latest)](https://elc-flow-engine.readthedocs.io/en/latest/?badge=latest)

 
 ## 介绍
 ELC使用
  
 ## 指令
 ### 生成文档
  > sphinx-apidoc -o source ../elcflow/ && make html
 ### 测试
  > pytest --cov=elcflow
  
 ## 使用
 ### 注册算子
 使用`register_elc_function`来注册是一个算子,算子包含以下属性:
 - name
    - 唯一的标识符,用来找到这个算子
 - outputs
    - list
    - 给输出取名字(在图上徐)
 - inputs
    - 可选
    - 定义输入的名称
    - 如果没有则使用函数中定义的名称
```python
@register_elc_function(outputs=['mul_result'], name='elc_mul')
def elc_mul(a, b):
    return a * b
```