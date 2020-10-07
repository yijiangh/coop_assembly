import os
import json
import datetime
import copy
from termcolor import cprint
from pybullet_planning import connect, LockRenderer, get_date
from .visualization import set_camera

from coop_assembly.data_structure import BarStructure, OverallStructure
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from .visualization import GROUND_COLOR, BACKGROUND_COLOR, SHADOWS

PICKNPLACE_DIRECTORY = os.path.join('..', '..', '..', 'tests', 'test_data')
PICKNPLACE_FILENAMES = {
    '12_bars' : '12_bars_point2triangle.json',
    'single_tet' : 'single_tet_point2triangle.json',
    # just an alias, I always mistype...
    '1_tets.json' : 'single_tet_point2triangle.json',
}

RESULTS_DIRECTORY = os.path.join('..', '..', '..', 'tests', 'results')

def get_assembly_path(assembly_name, file_dir=PICKNPLACE_DIRECTORY):
    if assembly_name.endswith('.json'):
        filename = os.path.basename(assembly_name)
    elif assembly_name in PICKNPLACE_FILENAMES:
        filename = PICKNPLACE_FILENAMES[assembly_name]
    else:
        filename = '{}.json'.format(assembly_name)
    root_directory = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root_directory, file_dir, filename))

def load_structure(test_file_name, viewer, color=(1,0,0,0)):
    """connect pybullet env and load the bar system
    """
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with LockRenderer():
        b_struct_data, o_struct_data = parse_saved_structure_data(get_assembly_path(test_file_name))
        if 'data' in b_struct_data:
            b_struct_data = b_struct_data['data']
        b_struct = BarStructure.from_data(b_struct_data)
        b_struct.create_pb_bodies(color=color)
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

def save_plan(problem, algorithm, trajectories, TCP_link_name=None, overwrite=True, element_from_index=None, suffix=None):
    here = os.path.dirname(__file__)
    plan_path = '{}_{}{}_solution{}.json'.format(problem, algorithm, '' if suffix is None else '_' + suffix, '' if overwrite else '_'+get_date())
    save_path = os.path.join(here, RESULTS_DIRECTORY, plan_path)
    with open(save_path, 'w') as f:
        data = {'problem' : problem,
                'write_time' : str(datetime.datetime.now()),
                # 'plan' : [jsonpickle.encode(p, keys=True) for p in trajectories]}
                # 'plan' : [p.to_data() for p in trajectories]}
                }
        data['plan'] = []
        e_path = []
        e_id = trajectories[0].element
        for traj in trajectories:
            print(traj)
            if traj.element is not None and e_id != traj.element:
                print('break')
                # break subprocess if there is a discontinuity in the element id
                data['plan'].append(copy.deepcopy(e_path))
                e_path = []
                e_id = traj.element
            tdata = traj.to_data()
            if TCP_link_name is not None:
                tdata.update({'link_path' : {TCP_link_name : traj.get_link_path(TCP_link_name)}})
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

        json.dump(data, f)
    cprint('Result saved to: {}'.format(save_path), 'green')

def parse_plan(file_name):
    here = os.path.dirname(__file__)
    save_path = os.path.join(here, RESULTS_DIRECTORY, file_name)
    with open(save_path) as json_file:
        data = json.load(json_file)
    cprint('Saved path parsed: file name:{} | write_time: {}'.format(
        data['problem'], data['write_time']), 'green')
    return data['plan']
