from typing import Union


def get_max_sep(L: int, max_sep: Union[None, int, float]) -> int:
    if max_sep is None:
        max_sep = L
    elif isinstance(max_sep, int):
        pass
    elif isinstance(max_sep, float):
        max_sep = int(L*max_sep)
    else:
        raise TypeError(max_sep.__class__)
    return max_sep