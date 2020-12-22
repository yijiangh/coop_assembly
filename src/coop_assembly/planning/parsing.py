import os
import json
import datetime
import copy
from termcolor import cprint
from collections import OrderedDict
from pybullet_planning import connect, LockRenderer, get_date, is_connected, RED
from .visualization import set_camera

from coop_assembly.data_structure import BarStructure, OverallStructure
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions.shared_const import METER_SCALE
from .visualization import GROUND_COLOR, BACKGROUND_COLOR, SHADOWS

# Configuration = namedtuple('Configuration', ['seed', 'problem', 'algorithm', 'bias', 'max_time',
#                                              'cfree', 'disable', 'stiffness', 'motions', 'ee_only'])
class Config(object):
    def __init__(self, args):
        self.problem = args.problem
        self.args = args

    def to_data(self):
        data = {}
        data['bar_only'] = bool(self.args.bar_only)
        data['stiffness'] = bool(self.args.stiffness)
        data['collision'] = bool(self.args.collisions)
        data['teleops'] = bool(self.args.teleops)
        data['partial_ordering'] = bool(self.args.partial_ordering)
        data['chosen_bars'] = [int(b) for b in self.args.subset_bars] if self.args.subset_bars is not None else None
        return data

PICKNPLACE_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'test_data')
PICKNPLACE_FILENAMES = {
    '12_bars' : '12_bars_point2triangle.json',
    'single_tet' : 'single_tet_point2triangle.json',
    # just an alias, I always mistype...
    '1_tets.json' : 'single_tet_point2triangle.json',
}


HERE = os.path.dirname(__file__)
RESULTS_DIRECTORY = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'tests', 'results'))

def get_assembly_path(assembly_name, file_dir=PICKNPLACE_DIRECTORY):
    if assembly_name.endswith('.json'):
        filename = os.path.basename(assembly_name)
    elif assembly_name in PICKNPLACE_FILENAMES:
        filename = PICKNPLACE_FILENAMES[assembly_name]
    else:
        filename = '{}.json'.format(assembly_name)
    model_path = os.path.abspath(os.path.join(file_dir, filename))
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return model_path

def load_structure(test_file_name, viewer=False, color=(1,0,0,0)):
    """connect pybullet env and load the bar system
    """
    if not is_connected():
        connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with LockRenderer():
        b_struct_data, o_struct_data = parse_saved_structure_data(get_assembly_path(test_file_name))
        if 'data' in b_struct_data:
            b_struct_data = b_struct_data['data']
        b_struct = BarStructure.from_data(b_struct_data)
        b_struct.name = test_file_name
        b_struct.get_element_from_index(color=color, regenerate=True)
        o_struct = None
        if o_struct_data is not None:
            if 'data' in o_struct_data:
                o_struct_data = o_struct_data['data']
            o_struct = OverallStructure.from_data(o_struct_data)
            o_struct.struct_bar = b_struct # TODO: better way to do this
        # set_camera([attr['point_xyz'] for v, attr in o_struct.nodes(True)])
        endpts_from_element = b_struct.get_axis_pts_from_element(scale=1e-3)
        set_camera([p[0] for e, p in endpts_from_element.items()], scale=1.)
    return b_struct, o_struct

def unpack_structure(bar_struct, chosen_bars=None, scale=METER_SCALE, color=RED):
    """extract geometric info from a BarStructure instance

    Parameters
    ----------
    bar_struct : [type]
        [description]
    chosen_bars : [type], optional
        [description], by default None
    scale : [type], optional
        [description], by default METER_SCALE

    Returns
    -------
    element_from_index : dict
        element index => coop_assembly.data_structure.utils.Element
    grounded_elements : list
        grounded element indices
    contact_from_connectors : dict
        ((elem 1, elem 2)) => (contact line pt 1, pt 2)
    connectors : list
        contact keys (element id pairs)
    """
    element_from_index = bar_struct.get_element_from_index(indices=chosen_bars, scale=scale, color=color)
    grounded_elements = bar_struct.get_grounded_bar_keys(indices=chosen_bars)
    contact_from_connectors = bar_struct.get_connectors(indices=chosen_bars, scale=scale)
    connectors = list(contact_from_connectors.keys())
    return element_from_index, grounded_elements, contact_from_connectors, connectors

def save_plan(config, trajectories, save_link_names=None, overwrite=True, element_from_index=None, bar_struct=None,
    suffix=None, extra_data=None):
    plan_path = '{}_{}-{}{}solution{}.json'.format(config.problem.split('.json')[0], config.args.algorithm, config.args.bias,
        '' if suffix is None else '_' + suffix, '' if overwrite else '_'+get_date())
    save_path = os.path.join(RESULTS_DIRECTORY, plan_path)

    config_data = config.to_data()
    from .logger import get_global_parameters
    with open(save_path, 'w') as f:
        data = OrderedDict()
        data['problem'] = config.problem,
        data['config'] = config_data,
        data['write_time'] = str(datetime.datetime.now()),
        data['parameters'] = get_global_parameters(),
        data['plan'] = []
        e_path = []
        e_id = trajectories[0].element
        for traj in trajectories:
            # print(traj)
            if traj.element is not None and e_id != traj.element:
                print('break')
                # break subprocess if there is a discontinuity in the element id
                data['plan'].append(copy.deepcopy(e_path))
                e_path = []
                e_id = traj.element
            tdata = traj.to_data()
            if save_link_names is not None:
                link_path_data = {link_name : traj.get_link_path(link_name) for link_name in save_link_names}
                tdata.update({'link_path' : link_path_data})
            e_path.append(tdata)
        else:
            data['plan'].append(e_path)

        if element_from_index is not None:
            element_data = {}
            for e_id, element in element_from_index.items():
                element_data[e_id] = {
                    'axis_endpoints' : [list(pt) for pt in element.axis_endpoints],
                    'radius' : element.radius,
                    'goal_pose' : element.goal_pose.to_data(),
                }
            data['element_from_index'] = element_data

        if bar_struct:
            from .stiffness import conmech_model_from_bar_structure
            chosen_bars = config_data['chosen_bars'] if config_data['chosen_bars'] and len(config_data['chosen_bars']) > 0 else None
            model, fem_element_from_bar_id = conmech_model_from_bar_structure(bar_struct, chosen_bars=chosen_bars)
            model_data = model.to_data()
            model_data['fem_element_from_bar_id'] = {bar : list(fem_es) for bar, fem_es in fem_element_from_bar_id.items()}
            data['conmech_model'] = model_data

        if extra_data is not None:
            data.update(extra_data)

        json.dump(data, f)
    cprint('Result saved to: {}'.format(os.path.abspath(save_path)), 'green')

##############################################

def parse_plan(file_name):
    save_path = os.path.join(RESULTS_DIRECTORY, file_name)
    with open(save_path) as json_file:
        data = json.load(json_file)
    cprint('Saved path parsed: problem:{} | write_time: {}'.format(
        data['problem'], data['write_time']), 'green')
    # return data['plan']
    return data
