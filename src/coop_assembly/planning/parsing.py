import os
from pybullet_planning import connect, LockRenderer
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

def get_assembly_path(assembly_name):
    if assembly_name.endswith('.json'):
        filename = os.path.basename(assembly_name)
    elif assembly_name in PICKNPLACE_FILENAMES:
        filename = PICKNPLACE_FILENAMES[assembly_name]
    else:
        filename = '{}.json'.format(assembly_name)
    root_directory = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root_directory, PICKNPLACE_DIRECTORY, filename))

def load_structure(test_file_name, viewer, color=(1,0,0,0)):
    """connect pybullet env and load the bar system
    """
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with LockRenderer():
        b_struct_data, o_struct_data, _ = parse_saved_structure_data(get_assembly_path(test_file_name))
        o_struct = OverallStructure.from_data(o_struct_data)
        b_struct = BarStructure.from_data(b_struct_data)
        b_struct.create_pb_bodies(color=color)
        o_struct.struct_bar = b_struct # TODO: better way to do this
        set_camera([attr['point_xyz'] for v, attr in o_struct.nodes(True)])
    return b_struct, o_struct

