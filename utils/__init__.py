class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AnyKey:
    def __init__(self, d):
        self.d = d

    def __getitem__(self, item):
        return self.d


def dict_flatten(d, parent_key='', sep='.'):
    """
    将嵌套字典展平为单层字典。

    :param d: dict, 要展平的嵌套字典
    :param parent_key: str, 父级键前缀（递归时使用）
    :param sep: str, 键路径的分隔符
    :return: dict, 展平后的字典
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):  # 如果值是字典，递归展平
            items.update(dict_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def dict_unflatten(flat_dict, sep='.'):
    """
    将展平的单层字典还原为嵌套字典。

    :param flat_dict: dict, 展平的单层字典
    :param sep: str, 键路径的分隔符
    :return: dict, 嵌套还原后的字典
    """
    nested_dict = {}
    for flat_key, value in flat_dict.items():
        keys = flat_key.split(sep)  # 根据分隔符拆分键路径
        current = nested_dict
        for key in keys[:-1]:  # 遍历键路径的前部分，创建嵌套结构
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value  # 设置最终的值
    return nested_dict
