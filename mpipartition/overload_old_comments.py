from partition import Partition
from typing import Mapping, Tuple, Union
import numpy as np

ParticleDataT = Mapping[str, np.ndarray]

def overload(
    partition: Partition,
    box_size: float,
    data: ParticleDataT,
    overload_length: float,
    coordinate_keys: Tuple[str, ...],   # AC - these ellipses make the tuple have variable length and homogeneous type
    *,
    verbose: Union[bool, int] = False,
):
    """Copy data within an overload length to the 26 neighboring ranks      # AC - change to 8 or 26

    This method assumes that the volume cube is periodic and will wrap the data
    around the boundary interfaces.

    Parameters
    ----------
    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    box_size:
        the size of the full volume cube

    data:
        The treenode / coretree data that should be distributed     # AC - datatype is "Mapping[str, numpy.ndarray]"  (sounds kind of like a dictionary)

    overload_length:
        The thickness of the boundary layer that will be copied to the
        neighboring rank

    coordinate_keys:
        The columns in `data` that define the position of the object

    verbose:
        If True, print summary statistics of the distribute. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    Returns
    -------
    data: TreeDataT
        The combined data of objects within the rank's subvolume as well as the
        objects within the overload region of neighboring ranks

    Notes
    -----

    The function does not change the objects' coordinates or alter any data.
    Objects that have been overloaded accross the periodic boundaries will still
    have the original positions. In case "local" coordinates are required, this
    will need to be done manually after calling this function.

    """
    nranks = partition.nranks
    if nranks == 1:
        return data

    rank = partition.rank
    comm = partition.comm
    origin = box_size * np.array(partition.origin)         # AC - origin for that processor aka subvolume (note: no particles here)
    extent = box_size * np.array(partition.extent)         # AC - Length along each axis of this processors subvolume
    neighbors = partition.neighbors                        # AC - the neighboring ranks of a given process

    # Find all overload regions each particle should be in
    overload = {}                                          # AC - a set (in the making) to hold ndim masks over the data. Each mask is the shape of data[coordinate_keys[ndim]], so it can hold the positions of all particles along that coordinate axis. We will then mask over these lists of coordinates to indicate which particles inhabit boundary regions of their rank's subvolume. Think of this set as having shape `(ndims, nparticles)`
    # AC - below, I say "near" and "far" to indicate relative distance from boundary region to the origin
    for (
        i,                                                  # AC - why is this typeset this way?
        x,
    ) in enumerate(coordinate_keys):                        # AC - repeat for each dimension
        _i = np.zeros_like(data[x], dtype=np.int8)          # AC - positions of particles not inhabiting boundary regions are marked 0
        _i[data[x] < origin[i] + overload_length] = -1      # AC - mark positions of particles inhabiting the "near" boundary region -1
        _i[data[x] > origin[i] + extent[i] - overload_length] = 1     # AC - mark particles inhabiting the "far" boundary region with 1
        overload[i] = _i                                    # AC - overload becomes the mask over all particle positions

    print("Overload for rank ")
    # Get particle indices of each of the 27 neighbors overload
    exchange_indices = [np.empty(0, dtype=np.int64) for i in range(nranks)]      # AC - make 27 empty arrays of length zero. These will hold the positions of particles inhabiting the boundary regions of each rank

    def add_exchange_indices(mask, i, j, k):
        n = neighbors[i + 1, j + 1, k + 1]      # AC - returns the rank of the neighboring process
        if n != rank:                           # AC - make sure we're not on the same rank as the neighbors we are examining
            exchange_indices[n] = np.union1d(exchange_indices[n], np.nonzero(mask)[0])   # AC - add positions of particles in the boundary regions to the `exchange_indices` array for this rank. The union means we will not double count particles that inhabit multiple boundary regions in the same nodes (e.g. particles in corner regions, which necessarily inhabit multiple edge and face regions as well)

    # AC - There will be 8 unique rounds here
    for i in [-1, 1]:
        # face
        maski = overload[0] == i              # AC - mask over the face boundary regions (where first dimension of overload is marked -1 or 1)
        add_exchange_indices(maski, i, 0, 0)  # AC - this function is called once for each unique face, edge, and corner region in each rank

        for j in [-1, 1]:
            # edge
            maskj = maski & (overload[1] == j)     # AC - mask over edge boundary regions (where both first and second dims are marked -1 or 1)
            add_exchange_indices(maskj, i, j, 0)

            for k in [-1, 1]:
                # corner
                maskk = maskj & (overload[2] == k) # AC - mask over corner boundary regions (all dims are marked -1 or 1)
                add_exchange_indices(maskk, i, j, k)

        for k in [-1, 1]:                          # AC - mask over edge boundary regions (where both first and third dims are marked -1 or 1)
            # edge
            maskk = maski & (overload[2] == k)
            add_exchange_indices(maskk, i, 0, k)

    for j in [-1, 1]:                         # AC - mask over face boundary regions (where second dim is marked -1 or 1)
        # face
        maskj = overload[1] == j
        add_exchange_indices(maskj, 0, j, 0)

        for k in [-1, 1]:                     # AC - mask over edge boundary regions (where both second and third dims are marked -1 or 1)
            # edge
            maskk = maskj & (overload[2] == k)
            add_exchange_indices(maskk, 0, j, k)

    for k in [-1, 1]:                         # AC - mask over face boundary regions (where third dim is marked -1 or 1)
        # face
        maskk = overload[2] == k
        add_exchange_indices(maskk, 0, 0, k)

    # Check how many elements will be sent
    send_counts = np.array([len(i) for i in exchange_indices], dtype=np.int32)
    print("For rank ", rank, " send_counts is: ", send_counts)
    send_idx = np.concatenate(exchange_indices)
    send_displacements = np.insert(np.cumsum(send_counts)[:-1], 0, 0)
    print("And send_displacements is: ", send_displacements)
    total_to_send = np.sum(send_counts)

    # Check how many elements will be received
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    print("recv_counts is: ", recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    print("recv_displacements: ", recv_displacements)
    total_to_receive = np.sum(recv_counts)

    # debug message
    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Overload Debug Rank {i}")
                print(f" - rank sends    {total_to_send:10d} particles")
                print(f" - rank receives {total_to_receive:10d} particles")
                print(f" - send_counts:        {send_counts}")
                print(f" - send_displacements: {send_displacements}")
                print(f" - recv_counts:        {recv_counts}")
                print(f" - recv_displacements: {recv_displacements}")
                print(f" - overload_x: {overload[0]}")
                print(f" - overload_y: {overload[1]}")
                print(f" - overload_z: {overload[2]}")
                print(f" - send_idx: {send_idx}")
                print(f"", flush=True)
            comm.Barrier()

    # send data all-to-all, each array individually
    data_new = {}

    for k in data.keys():
        # prepare send-array
        ds = data[k][send_idx]
        # prepare recv-array
        dr = np.empty(total_to_receive, dtype=ds.dtype)
        # exchange data
        s_msg = [ds, (send_counts, send_displacements), ds.dtype.char]
        r_msg = [dr, (recv_counts, recv_displacements), ds.dtype.char]
        comm.Alltoallv(s_msg, r_msg)
        # add received data to original data
        data_new[k] = np.concatenate((data[k], dr))

    return data_new
