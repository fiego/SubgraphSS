
from scipy.spatial.distance import cdist

def scipy_cosine(a_vec, b_vec):
    cosine = 1. - cdist(a_vec, b_vec, 'cosine')
    return cosine

def cosine_similarity(a_vec, b_vec):
    """ 计算两个向量余弦值
    Arguments:
        a_vec {[type]} -- a 向量
        b_vec {[type]} -- b 向量
    Returns:
        [type] -- [description]
    """
    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = 0.0
    for a, b in zip(a_vec, b_vec):
        dot_val += a * b
        a_norm += a ** 2
        b_norm += b ** 2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm * b_norm) ** 0.5)
    return cos