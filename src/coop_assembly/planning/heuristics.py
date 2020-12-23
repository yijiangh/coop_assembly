from pybullet_planning import RED, apply_alpha, INF

from coop_assembly.help_functions.shared_const import METER_SCALE
from coop_assembly.planning.utils import compute_z_distance
from coop_assembly.planning.parsing import unpack_structure
from coop_assembly.planning.stiffness import plan_stiffness

DISTANCE_HEURISTICS = [
    'z',
    # 'dijkstra',
    #'online-dijkstra',
    'plan-stiffness', # TODO: recategorize
]

HEURISTICS = ['none'] + DISTANCE_HEURISTICS #  + STIFFNESS_HEURISTICS

def get_heuristic_fn(robot, bar_struct, heuristic, forward=False, checker=None, fem_element_from_bar_id=None):
    sign = +1 if forward else -1
    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, scale=METER_SCALE, color=apply_alpha(RED,0.2))
    all_elements = frozenset(element_from_index.keys())

    # precomputed heuristics
    order = None
    if heuristic == 'plan-stiffness':
        order = plan_stiffness(bar_struct, all_elements, initial_position=None, checker=checker, fem_element_from_bar_id=fem_element_from_bar_id,
            stiffness=True, heuristic='z', max_time=INF, max_backtrack=0)

    def h_fn(printed, element): # , position, conf
        # ! Queue minimizes the statistic
        # forward adds, backward removes
        # structure = printed | {element} if forward else printed - {element}

        # TODO: weighted average to balance cost and bias
        if heuristic == 'none':
            return 0
        elif heuristic == 'z':
            return sign * compute_z_distance(element_from_index, element)
        elif heuristic == 'plan-stiffness':
            if order is None:
                return None
            return sign*order[element]
        raise ValueError(heuristic)

    return h_fn
