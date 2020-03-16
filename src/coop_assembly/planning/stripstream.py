from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.algorithms.focused import solve_focused #, CURRENT_STREAM_PLAN
from pddlstream.algorithms.disabled import process_stream_plan
from pddlstream.language.constants import And, PDDLProblem, print_solution, DurativeAction, Equal
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.language.function import FunctionInfo
from pddlstream.utils import read, get_file_path, inclusive_range
from pddlstream.language.temporal import compute_duration, get_end

##################################################

def get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes,
                   trajectories=[], temporal=True, local=False, **kwargs):
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################
