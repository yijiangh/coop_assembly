import os
import numpy as np
import pybullet_planning as pp

from coop_assembly.data_structure import BarStructure, GROUND_INDEX
from coop_assembly.help_functions.helpers_geometry import find_points_extreme, calculate_coord_sys, compute_local_coordinate_system, \
    bar_sec_verts, compute_contact_line_between_bars
from coop_assembly.geometry_generation.utils import get_element_neighbors
from coop_assembly.planning.visualization import draw_element, GROUND_COLOR, BACKGROUND_COLOR, SHADOWS, set_camera, \
    label_points, check_model
from coop_assembly.help_functions.parsing import export_structure_data

from beam_assembly import BeamAssembly
from multi_tangent import DATA_DIRECTORY

##################################

