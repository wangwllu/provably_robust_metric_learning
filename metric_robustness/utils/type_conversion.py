

def str_to_int_list(s):
    if s is None:
        return None

    return [int(a) for a in s.split(',')]
