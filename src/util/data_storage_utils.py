import numpy.typing as npt

def np_to_list(np_arr: npt.ArrayLike):
    try:
        return np_arr.tolist()
    except AttributeError:
        return np_arr