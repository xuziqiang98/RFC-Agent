import logging
import sys
from pathlib import Path


class Logger(logging.Logger):
    """自定义日志记录器类，继承自标准的 logging.Logger。

    该日志记录器提供控制台和文件两种输出方式，具有格式化的输出功能。
    控制台输出定向到标准输出（stdout），而文件输出（如果启用）则写入
    指定目录下以记录器名称命名的日志文件中。

    Args:
        name: 日志记录器的名称。
        level: 日志记录级别，可以是整数或字符串。默认为 logging.INFO。
        logs_dir: 日志文件存储目录的路径。如果为 None，则禁用文件日志记录。
            默认为 None。

    Attributes:
        handlers: 已添加的日志处理器列表。
    """
    def __init__(
        self,
        name: str,
        level: int | str = logging.INFO,
        logs_dir: Path = None,
    ) -> None:
        super().__init__(name, level=level)

        formatter = logging.Formatter(
            fmt='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(formatter)
        s_handler.setLevel(level)
        self.addHandler(s_handler)

        if logs_dir is not None:
            logs_dir.mkdir(exist_ok=True)
            log_file = logs_dir / f'{name}.log'
            f_handler = logging.FileHandler(log_file, 'a')
            f_handler.setFormatter(formatter)
            f_handler.setLevel(level)
            self.addHandler(f_handler)

    def set_level(self, level: int | str) -> None:
        """设置日志记录器和所有处理器的日志级别。

        Args:
            level: 新的日志级别，可以是整数或字符串。
        """
        self.setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)


class NullLogger:
    """空日志记录器的实现，采用空对象模式。

    该类实现了一个空日志记录器，会静默丢弃所有日志记录调用。
    当不需要实际的日志记录时，可以用作真实日志记录器的替代品，
    无需修改调用代码。

    Example:
        logger = NullLogger()
        logger.info("此消息将被丢弃")  # 无输出
        logger.error("此错误也将被丢弃")  # 无输出
    """
    def __getattr__(self, name):
        """捕获并忽略所有方法调用。

        Args:
            name: 被调用的方法名称。

        Returns:
            一个空函数，接受任意参数但不执行任何操作。
        """
        return lambda *args, **kwargs: None
