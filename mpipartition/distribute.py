from .partition import Partition, MPI
from typing import Mapping, Tuple, Union
import sys
import numpy as np

ParticleDataT = Mapping[str, np.ndarray]


def distribute(
    partition: Partition,
    box_size: float,
    data: ParticleDataT,
    xyz_keys: Tuple[str, str, str],    # AC - replace this with `Tuple[str, ...]`, the ellipses should annotate a tuple of variable length and homogeneous type
    *,
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
) -> ParticleDataT:
    """Distribute data among MPI ranks according to data position and volume partition

    The position of each TreeData element is given by the x, y, and z columns
    specified with `xyz_keys`.

    Parameters
    ----------

    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    box_size:
        The size of the full simulation volume

    data:
        The treenode / coretree data that should be distributed

    xyz_keys:                                        # AC - might want to change this name so it's more general (e.g. for 2D cases?)
        The columns in `data` that define the position of the object

    verbose:
        If True, print summary statistics of the distribute. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    verify_count:
        If True, make sure that total number of objects is conserved

    Returns
    -------
    data: ParticleDataT
        The distributed treenode / coretree data (i.e. the data that this rank
        owns)

    """
    # get some MPI and partition parameters
    nranks = partition.nranks
    if nranks == 1:
        return data

    rank = partition.rank
    comm = partition.comm
    ranklist = np.array(partition.ranklist)
    extent = box_size * np.array(partition.extent)

    # count number of particles we have
    total_to_send = len(data[xyz_keys[0]])             # AC - this is just a way to get len(data['x'])

    if total_to_send > 0:
        #home_idx = np.zeros([])     # AC - could find home of each particle in the for loop instead?
        # Check validity of coordinates
        for i in range(3):           # AC - Replace "3" with the variable `len(xyz_keys)`
            _x = data[xyz_keys[i]]
            _min = _x.min()
            _max = _x.max()
            if _min < 0 or _max > box_size:
                print(
                    f"Error in distribute: position {xyz_keys[i]} out of range: [{_min}, {_max}]",
                    file=sys.stderr,
                    flush=True,
                )
                comm.Abort()

        # Find home of each particle
        _i = (data[xyz_keys[0]] / extent[0]).astype(np.int32)
        _j = (data[xyz_keys[1]] / extent[1]).astype(np.int32)
        _k = (data[xyz_keys[2]] / extent[2]).astype(np.int32)

        _i = np.clip(_i, 0, partition.decomp[0] - 1)
        _j = np.clip(_j, 0, partition.decomp[1] - 1)
        _k = np.clip(_k, 0, partition.decomp[2] - 1)
        home_idx = ranklist[_i, _j, _k]
    else:
        home_idx = np.empty(0, dtype=np.int32)

    # sort by rank
    s = np.argsort(home_idx)
    home_idx = home_idx[s]

    # offsets and counts
    send_displacements = np.searchsorted(home_idx, np.arange(nranks))
    send_displacements = send_displacements.astype(np.int32)
    send_counts = np.append(send_displacements[1:], total_to_send) - send_displacements
    send_counts = send_counts.astype(np.int32)

    # announce to each rank how many objects will be sent
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)

    # number of objects that this rank will receive
    total_to_receive = np.sum(recv_counts)

    # debug message
    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Distribute Debug Rank {i}")
                print(f" - rank has {total_to_send} particles")
                print(f" - rank receives {total_to_receive} particles")
                print(f" - send_counts:        {send_counts}")
                print(f" - send_displacements: {send_displacements}")
                print(f" - recv_counts:        {recv_counts}")
                print(f" - recv_displacements: {recv_displacements}")
                print(f"", flush=True)
            comm.Barrier()

    # send data all-to-all, each array individually
    data_new = {k: np.empty(total_to_receive, dtype=data[k].dtype) for k in data.keys()}

    for k in data.keys():
        d = data[k][s]
        s_msg = [d, (send_counts, send_displacements), d.dtype.char]
        r_msg = [data_new[k], (recv_counts, recv_displacements), d.dtype.char]
        comm.Alltoallv(s_msg, r_msg)

    if verify_count:
        local_counts = np.array(
            [len(data[xyz_keys[0]]), len(data_new[xyz_keys[0]])], dtype=np.int64
        )
        global_counts = np.empty_like(local_counts)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        if rank == 0 and global_counts[0] != global_counts[1]:
            print(
                f"Error in distribute: particle count during distribute was not maintained ({global_counts[0]} -> {global_counts[1]})",
                file=sys.stderr,
                flush=True,
            )
            comm.Abort()

    return data_new
