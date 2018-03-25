import numpy as np


def toOne(array, flag):
    """
    多クラス分類の問題を、二値分類の問題として取り扱うための変換関数
    flagで指定されたものと一致すれば、"[0, 1]"に変換し、
    不一致であれば"[1, 0]"に変換したリストを返す。
    """
    obj = np.zeros((array.shape[0], 2))
    for i, item in enumerate(array):
        if item[flag] == 1:
            obj[i][0] = 1
        else:
            obj[i][1] = 1
    return obj
