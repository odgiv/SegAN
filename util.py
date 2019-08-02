from scipy.spatial.distance import directed_hausdorff

def hausdorf_distance(a, b):
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])