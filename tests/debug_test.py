import argparse
import numpy as np
from test_planning import test_stream
from test_generation import test_contact_pts

def main(args):
    if args.test_module == 'stream':
        test_stream(args.viewer, args.problem, args.collision, args.bar_only, args.debug)
    elif args.test_module == 'drop_contact':
        test_contact_pts()
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_module')
    parser.add_argument('--problem', default='single_tet')
    parser.add_argument('-v', '--viewer', action='store_true', help='turn on pybullet viewer')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    parser.add_argument('--collision', action='store_false', help='disable collision checking')
    parser.add_argument('--bar_only', action='store_true', help='only planning motion for the bars')
    parser.add_argument('--watch', action='store_true', help='watch planned traj')

    args = parser.parse_args()
    main(args)
