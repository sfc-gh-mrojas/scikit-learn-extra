import numba
# Fast inner loop for CNN.
# Author: Jan-Oliver Joswig
# License: 3-clause BSD
#

from queue import Queue as cppqueue
import numpy as np



@numba.jit()
def check_similarity(a, sa,b,c):
    """Check if the CNN criterion is fullfilled

    Check if `a` and `b` have at least `c` common elements.  Faster than
    computing the intersection (say with `numpy.intersect1d`) and
    comparing its length to `c`.

    Args:
        a: 1D Array of integers (neighbor point indices) that
           supports the buffer protocol.
        b: 1D Array of integers (neighbor point indices) that
           supports the buffer protocol.
        sa: Length of `a`. Only received here because already computed.
        c: Check if `a` and `b` share this many elements.

    Returns:
        True (1) or False (0)
    """

    i=0
    j=0
    sb = b.shape[0]  # Control variables
    ai=None
    bj=None                 # Checked elements
    common = 0             # Common neighbors count

    if c == 0:
        return 1

    for i in range(sa):
        ai = a[i]
        for j in range(sb):
            bj = b[j]
            if ai == bj:
                # Check similarity and return/move on early
                common += 1
                if common == c:
                    return 1
                break
    return 0

@numba.jit()
def commonnn_inner(neighborhoods,labels,core_candidates,min_samples):

    init_point=0
    point=0
    member=0
    member_i=0
    m = 0
    n = neighborhoods.shape[0]
    neighbors = None
    neighbor_neighbors=None
    current = 0  # Cluster (start at 0; noise = -1)
    membercount=0       # Current cluster size
    q = cppqueue() # FIFO queue

    # BFS find connected components
    for init_point in range(n):
        # Loop over points and find source node
        if core_candidates[init_point] == 0:
            # Point already assigned or not enough neighbors
            continue
        core_candidates[init_point] = 0  # Mark point as included

        neighbors = neighborhoods[init_point]
        m = neighbors.shape[0]

        labels[init_point] = current     # Assign cluster label
        membercount = 1

        while True:
            for member_i in range(m):
                # Loop over connected neighboring points
                member = neighbors[member_i]
                if core_candidates[member] == 0:
                    # Point already assigned or not enough neighbors
                    continue

                neighbor_neighbors = neighborhoods[member]
                if check_similarity(             # Conditional growth
                        neighbors, m, neighbor_neighbors, min_samples):
                    core_candidates[member] = 0  # Point included
                    labels[member] = current     # Assign cluster label
                    membercount += 1             # Cluster grows
                    q.put(member)               # Add point to queue

            if q.empty():
                # No points left to check
                if membercount == 1:
                    # Cluster to small -> effectively noise
                    labels[init_point] = -1  # Revert assignment
                    current -= 1             # Revert cluster number
                break

            point = q.get()  # Get the next point from the queue
            if not q.empty():
                q.get()

            neighbors = neighborhoods[point]
            m = neighbors.shape[0]

        current += 1  # Increase cluster number
