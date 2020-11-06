import os
import json
import datetime
from collections import OrderedDict
from termcolor import cprint

def export_structure_data(bar_struct_data, overall_struct_data=None, save_dir='', file_name=None, indent=None, opt_data=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    #     raise ValueError('directory not exists: {}'.format(save_dir))

    data = OrderedDict()
    data['write_time'] = str(datetime.datetime.now())
    data['bar_structure'] = bar_struct_data
    data['overall_structure'] = overall_struct_data
    data['opt_data'] = opt_data

    file_name = file_name or 'coop_assembly_data.json'
    full_save_path = os.path.join(save_dir, file_name)

    with open(full_save_path, 'w') as f:
        json.dump(data, f, indent=indent)
    cprint('data saved to {}'.format(full_save_path), 'green')

def parse_saved_structure_data(file_path):
    with open(file_path, 'r')  as f:
        data = json.load(f)
    cprint('Parsed file name: {} | write_time: {} | '.format(file_path, data['write_time']), 'green')
    o_data = None
    if 'overall_structure' in data:
        o_data = data['overall_structure']
    return data['bar_structure'], o_data
