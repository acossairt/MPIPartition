from .partition import Partition, MPI
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
    ranklist = np.array(partition.ranklist)
    extent = box_size * np.array(partition.extent)

    # count number of particles we have
    total_to_send = len(data[coordinate_keys[0]])   # AC - this is just a way to get the number of particles
    
    if total_to_send > 0:
        
        coords_to_rank = []                         # AC - this list will hold lists of coordinates (one list of coordinates for each dimension)
        # Check validity of coordinates
        for i in range(len(coordinate_keys)):       # AC - Replaced "3" with the variable `len(coordinate_keys)`
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
            _i = (data[coordinate_keys[i]] / extent[i]).astype(np.int32)
            _i = np.clip(_i, 0, partition.decomp[i] - 1)
            coords_to_rank.append([_i])              # AC - add lists of coordinates, one dimension at a ti
            
            # AC - what does this look like right now?
            # I think coords_to_rank is a list of lists:
            # coords_to_rank = [ [x0, x1, x2,...], [y0, y1, y2,...], [z0, z1, z2,...]]
            # I want it to look like: 
            # coords_to_rank = [ (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), ...]

        coords_to_rank = zip(*coords_to_rank)
        home_idx = ranklist[tuple(coords_to_rank)]  # AC - this stores the "home rank" of each particle (i.e. which rank is doing the calculations for each particle)

    # sort by rank
    s = np.argsort(home_idx)
    home_idx = home_idx[s]   # AC - sort home_idx in order of the ranks (so all particles on rank 0 will be next to each other, same with rank 1, etc.)

    # offsets and counts
    send_displacements = np.searchsorted(home_idx, np.arange(nranks))   # AC - Find indices (in the sorted array home_idx) where elements (from np.arange(nranks), which should look like (0, 1, 2, ..., n)) should be inserted to maintain order.
    # AC - ooooh it seems like this is finding the "break-points" between lists of the same rank
    # AC - if something in nranks has the same value as an element in home_idx (which it certainly will), does it get inserted before or after that point?
    # AC - seems to be before (which is why we skip the first element of send_displacements; we know it will be zero)
    send_displacements = send_displacements.astype(np.int32)            # AC - make type `int`
    send_counts = np.append(send_displacements[1:], total_to_send) - send_displacements     # AC - append all of the send_displacements except the first one with total_to_send (which is the number of coordinates in the x direction, I think? Maybe number of particles?)
    # AC - so that is the list of indices of separation between ranklists (except the 0th one), and then we append the total number of particles (is that just one value?), and subtract from each of those the indices of separation between ranklists? (or are we subtracting those actual indices?)
    send_counts = send_counts.astype(np.int32)                          # AC - make type `int`

    # announce to each rank how many objects will be sent
    recv_counts = np.empty_like(send_counts)                # AC - empty array with same shape as `send_counts`
    comm.Alltoall(send_counts, recv_counts)                 # AC - hey look, it's some MPI!
    # scatter: Scatter data from one process to all other processes in a group
    # Allgather: Gather to All, gather data from all processes and distribute it to all other processes in a group
    # Alltoall: All to All Scatter/Gather, send data from all to all processes in a group
    # (Okay so like scatter, but from all ranks instead of just one. And then combine that with an Allgather)
    # SO says: Instead of providing a single value that should be shared with each other process, each process specifies 
    # one value to give to each other process
    # What we are sending: send_counts
    # Who will be receiving: recv_counts (that empty array we created) # Why isn't it ranks that are receiving? Or are we just "making a copy" of what is received?
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)    # AC - is this assuming 3 dimensions? Or is that third argument the axis?

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

    for k in data.keys():     # AC - does data.keys() make any assumptions about number of dimensions?
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
