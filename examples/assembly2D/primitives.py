import os, sys
from collections import namedtuple
import numpy as np

# def is_valid(p):
#     return np.greater_equal(p, [0, 0]) and np.greater([MAX_COLS, MAX_ROWS], p)


def get_length(vec, ord=1):
    return np.linalg.norm(vec, ord=ord)


def get_difference(p1, p2):
    return np.array(p2) - p1


def collision_test(p1, p2):
    # returns True if in collision
    return get_length(get_difference(p1, p2)) < 1e-3


def distance_fn(q1, q2):
    return get_length(get_difference(q1, q2))

##################################################

###############################################################

def get_wild_place_gen_fn(robots, obstacles, element_from_index, grounded_elements, partial_orders=[], collisions=True, \
    bar_only=False, initial_confs={}, teleops=False, fluent_special=False, **kwargs):
    """ fluent_special : True if we are running incremental + semantic attachment
    """
    gen_fn_from_robot = {}
    for robot in robots:
        ee_link = link_from_name(robot, EE_LINK_NAME) if not bar_only else get_links(robot)[-1]
        tool_link = link_from_name(robot, TOOL_LINK_NAME) if not bar_only else get_links(robot)[-1]
        # TODO end_effector is unused in bar_only setting
        end_effector = EndEffector(robot, ee_link=ee_link,
                                   tool_link=tool_link,
                                   visual=False, collision=True)

        # TODO: not need precompute_collisions when running incremental + semantic attachment
        # but just do it for now
        pick_gen_fn = get_place_gen_fn(end_effector, element_from_index, obstacles, verbose=False, \
            precompute_collisions=True, collisions=collisions, bar_only=bar_only, teleops=teleops, **kwargs)
        gen_fn_from_robot[robot] = pick_gen_fn

    def wild_gen_fn(robot_name, element, fluents=[]):
        robot = index_from_name(robots, robot_name)
        printed = []
        # print('E{} - fluent {}'.format(element, fluents))
        for fact in fluents:
            if fact[0] == 'assembled':
                if fact[1] != element:
                    printed.append(fact[1])
            else:
                raise NotImplementedError(fact[0])
        for command, in gen_fn_from_robot[robot](element, printed=printed):
            if not fluent_special:
                q1 = Conf(robot, np.array(command.start_conf), element)
                q2 = Conf(robot, np.array(command.end_conf), element)
                outputs = [(q1, q2, command)]
            else:
                outputs = [(command,)]
            facts = []
            yield WildOutput(outputs, facts)
            # facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
            # facts.append(('AtConf', robot_name, initial_confs[robot_name]))
            # cprint('print facts: {}'.format(command.colliding), 'yellow')
            # yield (q1, q2, command),

    return wild_gen_fn

def get_test_cfree():
    def test_fn(robot_name, traj, element):
        # return True if no collision detected
        return element not in traj.colliding
    return test_fn
