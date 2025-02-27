from pathlib import Path
import logging

from src.configs.config_base import ConfigBase


class PathConfig(ConfigBase):
    """管理和创建项目目录结构的配置类。

    该类继承自 ConfigBase，定义了项目的关键目录路径。
    它会自动创建所有已定义但不存在的目录。

    Args：
        root (Path): 项目根目录的路径
        src (Path): 源代码目录的路径
        data (Path): 数据目录的路径
        scripts (Path): 脚本目录的路径
        tests (Path): 测试目录的路径
        logs (Path): 日志目录的路径，位于数据目录下
    """
    root = Path(__file__).resolve().parents[2]
    src = root / 'src'
    data = root / 'data'
    scripts = root / 'scripts'
    tests = root / 'tests'
    logs = data / 'logs'

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class LoggerConfig(ConfigBase):
    """日志配置管理类。

    该类继承自 ConfigBase，用于管理和配置日志记录的相关参数。
    它定义了日志记录的基本设置，包括日志级别和日志文件存储目录。

    Args：
        level (int): 日志记录级别，默认为 logging.INFO
        logs_dir (Path): 日志文件存储目录的路径，默认使用 PathConfig 中定义的 logs 目录
    """
    level = logging.INFO
    logs_dir = PathConfig().logs

