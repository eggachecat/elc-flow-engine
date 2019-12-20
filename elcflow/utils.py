"""utils 包含日志等
"""
import logging
import re
import sys
from logging import Handler, Logger, StreamHandler


class LoggingMixin:
    """
    日志的Mixin
    """

    # def __init__(self, context=None):
    #     self._set_context(context)

    @property
    def log(self) -> Logger:
        try:
            # FIXME: LoggingMixin should have a default _log field.
            return self._log  # type: ignore
        except AttributeError:
            self._log = logging.root.getChild(
                self.__class__.__module__ + '.' + self.__class__.__name__
            )
            return self._log

    # def _set_context(self, context):
    #     if context is not None:
    #         set_context(self.log, context)

# def set_context(logger, value):
#     """
#     递归的设置log的handler的值: 例如log level
#     :param logger: logger
#     :param value: value to set
#     """
#     _logger = logger
#     while _logger:
#         for handler in _logger.handlers:
#             try:
#                 handler.set_context(value)
#             except AttributeError:
#                 pass
#         if _logger.propagate is True:
#             _logger = _logger.parent
#         else:
#             _logger = None
