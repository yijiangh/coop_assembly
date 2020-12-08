from .utils import compute_z_distance

DISTANCE_HEURISTICS = [
    'z',
    # 'dijkstra',
    #'online-dijkstra',
    # 'plan-stiffness', # TODO: recategorize
]

HEURISTICS = ['none'] + DISTANCE_HEURISTICS#  + STIFFNESS_HEURISTICS

def get_heuristic_fn(robot, element_from_index, heuristic, forward, checker=None):
    sign = +1 if forward else -1

    def h_fn(printed, element): # , position, conf
        # ! Queue minimizes the statistic
        # forward adds, backward removes
        structure = printed | {element} if forward else printed - {element}

        # TODO: weighted average to balance cost and bias
        if heuristic == 'none':
            return 0
        elif heuristic == 'z':
            return sign * compute_z_distance(element_from_index, element)
        raise ValueError(heuristic)

    return h_fn
