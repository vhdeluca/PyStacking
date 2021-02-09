def get_key(dict, value):
    """ Return the first key in the dictionary "dict" that contains the
    received value "value".

    Parameters
    ==========
    dict: Dict[Any, Any]
        Dictionary to be used.
    value: Any
        Value to be found in the dictionary.
    """
    return list(dict.keys())[list(dict.values()).index(value)]


def ds_exec_order(datasets):
    """ Return the execution order of all datasets in the stacking.

    Parameters
    ==========
    datasets: Dict[set, object]
        Dictionary with datasets.
    """
    sorted_keys = []
    dict_keys = {}

    # Build the list of sorted keys.
    for key, ds in datasets.items():
        key_sorted = tuple(sorted(key, key=lambda t: (t[0], t[1])))
        dict_keys[key_sorted] = key
        sorted_keys.append(key_sorted)

    # Sort the list of sorted keys.
    sorted_keys.sort(key=lambda t: tuple((x[0], x[1]) for x in t))

    # Return the sorted list with the original keys (not sorted).
    return [dict_keys[x] for x in sorted_keys]
