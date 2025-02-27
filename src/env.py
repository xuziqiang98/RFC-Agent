import os
from dotenv import load_dotenv

class Env:
    """环境变量管理类，实现了单例模式以确保全局配置的一致性。

    该类负责管理应用程序的环境变量和配置信息，通过单例模式确保在整个应用程序生命周期中
    只存在一个实例。它会自动加载 .env 文件中的配置，并将所有环境变量映射为实例属性。

    Features：
        1. 单例模式：通过重写 __new__ 方法确保类只有一个实例
        2. 自动加载：实例化时自动加载 .env 文件中的配置
        3. 属性映射：环境变量会被自动映射为实例的属性

    Examples：
        >>> env = Env()
        >>> print(env.OPENAI_API_KEY)  # 访问环境变量
        'your-api-key'
        >>> env2 = Env()  # 创建新实例
        >>> id(env) == id(env2)  # 验证单例模式
        True

    Attributes:
        _instance (Env): 类的单例实例，用于确保全局只有一个实例存在
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """创建或返回类的单例实例。

        重写 __new__ 方法以实现单例模式，确保类只有一个实例。首次创建实例时，
        会加载环境变量并将其映射为实例属性。

        Args:
            *args: 可变位置参数
            **kwargs: 可变关键字参数

        Returns:
            Env: 类的单例实例
        """
        if cls._instance is None:
            cls._instance = super(Env, cls).__new__(cls, *args, **kwargs)
            # 加载 .env 文件
            load_dotenv()
            # 将环境变量加载到实例属性中
            for key, value in os.environ.items():
                setattr(cls._instance, key, value)
        return cls._instance

    def __repr__(self):
        """返回实例的字符串表示。

        Returns:
            str: 包含所有环境变量的字符串表示
        """
        attrs = vars(self)
        attrs_str = ', '.join(f"{k}={v}" for k, v in attrs.items())
        return f"Env({attrs_str})"
