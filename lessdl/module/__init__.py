



def update_prev_state(state, sub_prefix, sub_state):
    """
    将子模块的状态sub_state, 更新到state, 每个sub_state的key变为sub_prefxi + key
    """
    assert state is not None
    for k, v in sub_state.items():
        state[sub_prefix + k] = v


def extract_sub_state(state, sub_prefix, pop=False):
    assert state is not None
    sub_state = {}
    start_idx = len(sub_prefix)
    for k, v in state.items():
        if k.startswith(sub_prefix):
            k = k[start_idx:]
            sub_state[k] = v
    if pop:
        for k in sub_state:
            state.pop(sub_prefix + k)
    return sub_state


