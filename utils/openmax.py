import numpy as np


def cdf(x, scale, shape):
    return 1-(np.exp(-(x/scale)**shape))


def get_cdf(A0_list, A1_list, outputs):
    dist_list = []
    dist = 0
    for A0, output in zip(A0_list, outputs):
        dist += (float(A0) - output)**2
    dist_list.append(round(dist, 4))
    
    dist = 0
    for A1, output in zip(A1_list, outputs):
        dist += (float(A1) - output)**2
    dist_list.append(round(dist, 4))

    scores = []

    scores.append(cdf(dist_list[0], scale=0.3609, shape=1.3325))
    scores.append(cdf(dist_list[1], scale=0.6948, shape=2.3505))

    return scores