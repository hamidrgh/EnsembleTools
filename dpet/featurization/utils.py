from typing import Union, List


def get_triu_indices(
        L: int,
        min_sep: int = 1,
        max_sep: Union[None, int, float] = None) -> List[list]:
    ids = [[], []]
    max_sep = get_max_sep(L=L, max_sep=max_sep)
    for i in range(L):
        for j in range(L):
            if i <= j:
                if j-i >= min_sep:
                    if j-i <= max_sep:
                        ids[0].append(i)
                        ids[1].append(j)
                continue
    return ids

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