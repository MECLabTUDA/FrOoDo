from collections import OrderedDict


def rename_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:]
        new_state_dict[new_key] = value
    return new_state_dict
