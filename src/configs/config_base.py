from typing import Any, Dict, Iterator


class ConfigBase:
    """提供类字典方式访问配置属性的基础配置类。

    该类作为配置管理的基础，允许通过属性方式和字典方式访问配置值。它会自动初始化
    类级别的属性，并支持通过 kwargs 动态更新属性。

    该类实现了字典类接口，包括迭代、键访问和键列表功能。

    Args：
        __dict__ (Dict[str, Any]): 存储所有配置属性的字典。
    """
    def __init__(self, **kwargs: Any) -> None:
        for name, value in self.__class__.__dict__.items():
            if not name.startswith("__") and not callable(value):
                setattr(self, name, value)
        self.__dict__.update(kwargs)
        self.__post_init__()

    def __repr__(self) -> str:
        attributes = ', '.join([f"{name}={value!r}" for name, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({attributes})"
    
    def __post_init__(self) -> None:
        pass

    def __iter__(self) -> Iterator:
        return iter(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def keys(self) -> Dict[str, Any].keys:
        return self.__dict__.keys()
