import sys

# small number epsilon
EPS = 1e-12

# large number infinite
INF = 1e23

# tolerance
TOL = 1e-3 # mm

# vertex correction, deciding how low a new vertex can go for a three-bar group
NODE_CORRECTION_TOP_DISTANCE = 80 # millimter

NODE_CORRECTION_SINE_ANGLE = 0.4

# unit used in coop_assembly converted to meter
METER_SCALE = 1e-3

# use box approx for pybullet
USE_BOX = False

def is_ironpython():
    return 'ironpython' in sys.version.lower()

IPY = is_ironpython()

# try:
import pybullet_planning
HAS_PYBULLET = True
# except ImportError:
    # print('Cannot import pybullet_plannig, related features disabled.')
    # HAS_PYBULLET = False

