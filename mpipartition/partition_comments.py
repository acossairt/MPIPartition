"""MPI Partitioning of a cube

"""

from mpi4py import MPI
import numpy as np
import sys, time
from typing import List

_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()
_nranks = _comm.Get_size()


def _factorize(n):
    i = 2
    factors = []
    while i <= n:
        if (n % i) == 0:
            factors.append(i)
            n /= i
        else:
            i = i + 1
    return factors


def _distribute_factors(factors, target):       # AC - Below, we do this with (nranks_factors, commensurate_topo)
    current_topo = np.ones_like(target)
    remaining_topo = np.copy(np.array(target))
    for f in factors[::-1]:
        commensurate = (remaining_topo % f) == 0
        if not np.any(commensurate):
            raise RuntimeError(
                "commensurate topology impossible with given rank number and target topo"
            )
        # add to lowest possible number
        s = np.argsort(current_topo)
        idx = s[np.nonzero(commensurate[s])[0][0]]
        current_topo[idx] *= f
        remaining_topo[idx] /= f
    return current_topo, remaining_topo


class Partition:
    """An MPI partition of a cubic volume

    Parameters
    ----------

    create_topo_unique : boolean
        If `True`, an additional graph communicator will be initialized
        connecting all (8 or 26) unique direct neighbors symmetrically

    mpi_waittime : float
        Time in seconds for which the initialization will wait, can fix certain
        MPI issues if ranks are not ready (e.g. `PG with index not found`)

    commensurate_topo : List[int]
        A proportional target topology for decomposition. When specified, a partition
        will be created so that `commensurate_topo[i] % partition.decomp[i] == 0` for
        all `i`. The code will raise a RuntimeError if such a decomposition is not
        possible.

    Examples
    --------

    Using Partition on 8 MPI ranks to split a periodic unit-cube

    >>> partition = Partition(1.0)
    >>> partition.rank
    0
    >>> partition.decomp
    np.ndarray([2, 2, 2])          # AC - This is slices (subvolumes) per dimension, and the indexing starts at 0 (so for 3D, this comes to 27)
    >>> partition.coordinates
    np.ndarray([0, 0, 0])
    >>> partition.origin
    np.ndarray([0., 0., 0.])
    >>> partition.extent
    np.ndarray([0.5, 0.5, 0.5])
    """

    def __init__(
        self,
        create_topo_unique: bool = False,      # AC - change everything that says 26 to "unique"
        mpi_waittime: float = 0,
        commensurate_topo: List[int] = None,
        ndims: int = 3,      # AC - this variable will tell us whether we are working in a box, a cube, or some other n-dimensional square
    ):
        self._rank = _rank
        self._nranks = _nranks
        if commensurate_topo is None:             # AC - what happens to self._decomp when coord (fed to self._topo) has fewer elements?
            self._decomp = MPI.Compute_dims(_nranks, np.zeros(ndims, dtype = np.int32))       # AC - MPI.Compute_dims is "a balanced distribution of processes per coordinate direction." For 3 dimensions and 8 ranks, this would be `(2, 2, 2)`. Note that this object is 1 dimensional (even if the field it is decomposing is n-dimensional)
        else:                                     
            nranks_factors = _factorize(self._nranks)
            decomp, remainder = _distribute_factors(nranks_factors, commensurate_topo)
            assert np.all(decomp * remainder == np.array(commensurate_topo))
            assert np.prod(decomp) == self._nranks
            self._decomp = decomp.tolist()                                

        periodic = [True,]*ndims     # AC - should account for various dimensions
        time.sleep(mpi_waittime)
        self._topo = _comm.Create_cart(self._decomp, periods=periodic)
        self._coords = list(self._topo.coords)
        time.sleep(mpi_waittime)
        self._neighbors = np.empty(np.array((3,)*ndims), dtype=np.int32)  # AC - Changed so that this can also be (3, 3)
        
        if ndims == 2: 
            # AC - Tried this, then cancelled it
            # coord = [(self._coords[x] + i) % self._decomp[x] for x in np.arange(ndims) for i in np.arange(ndims)] 
            
            for i in range(-1, 2):
                for j in range(-1, 2):
                        coord = [
                            (self._coords[0] + i) % self._decomp[0],
                            (self._coords[1] + j) % self._decomp[1],
                        ]
                        neigh = self._topo.Get_cart_rank(coord)     # AC - how will the new length of coord affect self._topo.Get_cart_rank?
                        self._neighbors[i + 1, j + 1] = neigh
                        # self._neighbors.append(neigh)
        elif ndims == 3:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        coord = [
                            (self._coords[0] + i) % self._decomp[0],
                            (self._coords[1] + j) % self._decomp[1],
                            (self._coords[2] + k) % self._decomp[2],
                        ]
                        neigh = self._topo.Get_cart_rank(coord)
                        self._neighbors[i + 1, j + 1, k + 1] = neigh
                        # self._neighbors.append(neigh)
                        
        else:
            print("Need to put an error and break here")

        self._extent = [1.0 / self._decomp[i] for i in range(ndims)]    # AC - changed 3 to ndims
        self._origin = [self._coords[i] * self._extent[i] for i in range(ndims)]     # AC - changed 3 to ndims

        # A graph topology linking all (8 or 26) unique neighbors
        self._topo_unique = None
        self._neighbors_unique = None
        self._nneighbors_unique = None
        if create_topo_unique:
            time.sleep(mpi_waittime)
            neighbors_unique = np.unique(
                np.array(
                    [n for n in self._neighbors.flatten() if n != self._rank],
                    dtype=np.int32,
                )
            )
            self._topo_unique = self._topo.Create_dist_graph_adjacent(
                sources=neighbors_unique, destinations=neighbors_unique, reorder=False
            )
            assert self._topo_unique.is_topo
            inout_neighbors_unique = self._topo_unique.inoutedges
            assert len(inout_neighbors_unique[0]) == len(inout_neighbors_unique[1])
            self._nneighbors_unique = len(inout_neighbors_unique[0])
            for i in range(self._nneighbors_unique):
                if inout_neighbors_unique[0][i] != inout_neighbors_unique[1][i]:
                    print(
                        "topo_unique: neighbors in sources and destinations are not ordered the same",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._topo.Abort()
            self._neighbors_unique = inout_neighbors_unique[0]

    def __del__(self):
        self._topo.Free()

    @property
    def comm(self):   # AC - interesting, so the communicator is... the topology?
        """3D Cartesian MPI Topology / Communicator"""
        return self._topo

    @property
    def comm_unique(self):
        """Graph MPI Topology / Communicator, connecting the neighboring ranks
        (symmetric)"""
        return self._topo_unique

    @property
    def rank(self):
        """int: the MPI rank of this processor"""
        return self._topo.rank

    @property
    def nranks(self):
        """int: the total number of processors"""
        return self._nranks

    @property
    def decomp(self):
        """np.ndarray: the decomposition of the cubic volume: number of ranks along each dimension"""
        return self._decomp

    @property
    def coordinates(self):
        """np.ndarray: 3D or 2D indices of this processor"""
        return self._coords

    @property
    def extent(self):
        """np.ndarray: Length along each axis of this processors subvolume (same for all procs)"""
        return self._extent

    @property
    def origin(self) -> np.ndarray:
        """np.ndarray: Cartesian coordinates of the origin of this processor"""
        return self._origin

    def get_neighbor(self, dx: int, dy: int, dz: int) -> int:
        """get the rank of the neighbor at relative position (dx, dy, dz)

        Parameters
        ----------

        dx, dy, dz: int
            relative position, one of `[-1, 0, 1]`
        """
        return self._neighbors[dx + 1, dy + 1, dz + 1]
    
    # AC - New function, just for 2D case (but maybe I could just make the first one more flexible?)
    def get_neighbor_2D(self, dx: int, dy: int) -> int:
        """get the rank of the neighbor at relative position (dx, dy)

        Parameters
        ----------

        dx, dy: int
            relative position, one of `[-1, 0, 1]`
        """
        return self._neighbors[dx + 1, dy + 1]

    @property
    def neighbors(self):
        """np.ndarray: a 3x3 or 3x3x3 array with the ranks of the neighboring processes
        (`neighbors[1,1,1]` is this processor)"""
        return self._neighbors

    @property
    def neighbors_unique(self):
        """np.ndarray: a flattened list of the unique neighboring ranks"""
        return self._neighbors_unique

    @property
    def neighbors_unique_count(self):
        """int: number of unique neighboring ranks"""
        return self._nneighbors_unique

    # AC - new version (Michael's solution)
    
    @property
    def ranklist(self):
        """np.ndarray: A complete list of ranks, aranged by their coordinates.
        The array has shape `partition.decomp`"""
        ranklist = np.empty(self.decomposition, dtype=np.int32)   # AC - Note: self.decomp is not giving its shape to be the shape of ranklist; self.decomp is giving itself to be the shape of ranklist. The shape of self.decomp is (ndims, ). The shape of ranklist is self.decomp.
        for idx in itertools.product(*map(range, self.decomposition)):
            ranklist[tuple(idx)] = self._topo.Get_cart_rank(idx)  # AC - that tuple() around idx might not be strictly necessary
        return ranklist

    
    
            