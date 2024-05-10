from typing import Union, List


def get_triu_indices(L: int, min_sep: int = 1, max_sep: Union[None, int, float] = None) -> List[list]:
    """
    Get the upper triangle indices of a square matrix with specified minimum and maximum separations.

    Parameters
    ----------
    L : int
        The size of the square matrix.
    min_sep : int, optional
        The minimum separation between indices. Default is 1.
    max_sep : Union[None, int, float], optional
        The maximum separation between indices. Default is None.

    Returns
    -------
    List[list]
        A list of lists containing the upper triangle indices of the square matrix.

    Notes
    -----
    This function returns the upper triangle indices of a square matrix with the specified minimum and maximum separations.
    """
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
    """
    Get the maximum separation between indices.

    Parameters
    ----------
    L : int
        The size of the matrix.
    max_sep : Union[None, int, float]
        The maximum separation between indices.

    Returns
    -------
    int
        The maximum separation between indices.

    Notes
    -----
    This function calculates the maximum separation between indices based on the size of the matrix and the provided maximum separation value.
    """
    if max_sep is None:
        max_sep = L
    elif isinstance(max_sep, int):
        pass
    elif isinstance(max_sep, float):
        max_sep = int(L*max_sep)
    else:
        raise TypeError(max_sep.__class__)
    return max_sep