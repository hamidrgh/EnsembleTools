import numpy as np

def unit_vectorize(a):
    """Convert an array with (*, N) angles in an array with (*, N, 2) sine and
    cosine values for the N angles."""
    v = np.concatenate([np.cos(a)[..., None], np.sin(a)[..., None]], axis=-1)
    return v

def unit_vector_distance(a0, a1):
    """Compute the sum of distances between two (*, N, 2) arrays storing the
    sine and cosine values of N angles."""
    v0 = unit_vectorize(a0)
    v1 = unit_vectorize(a1)
    # Distance between every pair of N angles.
    dist = np.sqrt(np.square(v0 - v1).sum(axis=-1))
    # We sum over the N angles.
    dist = dist.sum(axis=-1)
    return dist