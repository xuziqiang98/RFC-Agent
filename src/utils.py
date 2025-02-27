import inspect
from types import MethodType, FunctionType
from pathlib import Path
import random

def get_script_name() -> str:
    """获取调用此函数的脚本文件名（不含扩展名）。

    该函数使用 Python 的检查栈（inspection stack）来确定调用此函数的脚本。
    它获取调用者的文件名，并返回该路径的主干部分（不含扩展名的文件名）。

    Returns:
        str: 调用此函数的脚本文件名（不含扩展名）。
    """
    caller_frame_record = inspect.stack()[1]
    module_path = caller_frame_record.filename
    return Path(module_path).stem
