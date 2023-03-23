import matplotlib.pyplot as plt
from matplotlib import collections as plt_collections
from coop_assembly.data_structure import BarStructure, GROUND_INDEX
import numpy as np
from numpy.linalg import norm as npnorm

def plot_contact_distance(bar_struct: BarStructure):
    connectors = bar_struct.get_connectors()
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    # clean up duplicated bidirectional edges
    dist_data = dict()
    radius_lines = dict()
    count = 0
    for edge, c in connectors.items():
        n1, n2 = edge
        if n1 == GROUND_INDEX or n2 == GROUND_INDEX:
            continue
        reversed_edge = edge[::-1]
        if reversed_edge in dist_data:
            continue
        dist_data[edge] = (npnorm(np.array(c[0]) - np.array(c[1])))
        r1, r2 = bar_struct.node[n1]['radius'], bar_struct.node[n2]['radius']
        radius_lines[edge] = ([(count,r1+r2), (count+1,r1+r2)])
        count += 1

    # * histogram
    dist_list = list(dist_data.values())
    axes[0].hist(dist_list, 20, lw=2, ec="yellow", fc="green", alpha=0.5)
    axes[0].set_xlabel('Shortest distance between contact bars [m]')
    axes[0].set_ylabel('Count')

    # * line plot
    lc = plt_collections.LineCollection(list(radius_lines.values()), colors='blue', linewidths=2)
    axes[1].add_collection(lc)
    axes[1].plot(dist_list)
    axes[1].set_xlabel('Connector index')
    axes[1].set_ylabel('Dist between contact bars [m]')
    fig.suptitle('Contact dist from BarStructure')

    return fig