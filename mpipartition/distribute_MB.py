from partition import Partition, MPI
from typing import Mapping, Tuple, Union
import sys
import numpy as np

ParticleDataT = Mapping[str, np.ndarray]

# AC - I chose not to add an ndims parameter, and instead to get by with `coordinate_keys`
def distribute(
    partition: Partition,
    box_size: float,
    data: ParticleDataT,
    coordinate_keys: Tuple[str, ...],    # AC - renamed from `xyz_keys`. The ellipses annotate a tuple of variable length and homogeneous type
    *,
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
    dimensions: int = 3,
) -> ParticleDataT:
    """Distribute data among MPI ranks according to data position and volume partition

    The position of each TreeData element is given by its coordinates (e.g. x, y, and z in the 3D case)
    specified with `coordinate_keys`.

    Parameters
    ----------

    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    box_size:
        The size of the full simulation volume

    data:
        The treenode / coretree data that should be distributed     # AC - this object has type `Mapping[str, numpy.ndarray]`

    coordinate_keys:                                                
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
    ranklist = np.array(partition.ranklist)  # ranklist has shape self.decomp (so it will depend on the number of dimensions)
    print("The ranklist is: ", ranklist)
    extent = box_size * np.array(partition.extent)

    # count number of particles we have
    total_to_send = len(data[coordinate_keys[0]])   # AC - this is just a way to get the number of particles
    
    if total_to_send > 0:
        # Check validity of coordinates
        for i in range(dimensions):
            _x = data[coordinate_keys[i]]
            _min = _x.min()
            _max = _x.max()
            if _min < 0 or _max > box_size:
                print(
                    f"Error in distribute: position {coordinate_keys[i]} out of range: [{_min}, {_max}]",
                    file=sys.stderr,
                    flush=True,
                )
                comm.Abort()

        # Find home of each particle
        idx = np.array([data[coordinate_keys[i]] / extent[i] for i in range(dimensions)]).astype(np.int32)
        idx = np.clip(idx, 0, np.array(partition.decomp)[:, np.newaxis]-1)
        print("On its own, idx looks like: ", idx)
        print("As a tuple, idx looks like: ", tuple(idx))
        home_idx = ranklist[tuple(idx)]
    else:
        home_idx = np.empty(0, dtype=np.int32)
        
    print("TROUBLESHOOT")
    #print("home_idx looks like: ", home_idx)
    print(type(home_idx))
    print(type(home_idx[0]))
    # sort by rank
    s = np.argsort(home_idx)   # AC - added flatten?
    home_idx = home_idx[s]   # AC - sort home_idx in order of the ranks (so all particles on rank 0 will be next to each other, same with rank 1, etc.)
    

    # offsets and counts
    send_displacements = np.searchsorted(home_idx, np.arange(nranks))   # AC - Find the "break points" within the sorted list of home ranks
    send_displacements = send_displacements.astype(np.int32) 
    send_counts = np.append(send_displacements[1:], total_to_send) - send_displacements     # AC - Not sure what is happening here? We append all of the send_displacements except the first one with total_to_send (which is the number of particles, I think?) and then subtract from each of those the indices of separation between ranklists?
    send_counts = send_counts.astype(np.int32)

    # announce to each rank how many objects will be sent
    recv_counts = np.empty_like(send_counts)                # AC - empty buffer where we will put the output (helps memory allocation)
    comm.Alltoall(send_counts, recv_counts)                 # AC - send `send_counts` from all ranks to all other ranks, then gather all to all
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)    # AC - Not sure what is happening here?

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
            [len(data[coordinate_keys[0]]), len(data_new[coordinate_keys[0]])], dtype=np.int64
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
