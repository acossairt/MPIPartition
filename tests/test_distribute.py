#!/usr/bin/env python

"""Tests for `mpipartition` package."""

import sys
sys.path.append('/data/a/cpac/aurora/MPIPartition/mpipartition')
import pytest
from partition import Partition
from distribute import distribute
import numpy as np

def test_distribute(ndims, box_size, n_local = 1000):

    # partitioning a box with the available MPI ranks
    partition = Partition(ndims = ndims)#, create_topo_unique=True)
    rank = partition.rank
    nranks = partition.nranks
    origin = partition.origin
    extent = partition.extent
    print("Rank ", rank, " has origin ", origin, " and extent ", extent)
    
    # Create data according to desired dimensions
    possible_coords = "xyzwuv"
    coord_keys = possible_coords[:ndims]

    data = {
        x: np.random.uniform(0, 1*box_size, n_local) for i, x in enumerate(coord_keys)#,
        #'id': n_local * rank + np.arange(n_local)
    }

    # assign to rank by position
    data_distributed = distribute(partition, box_size, data, coord_keys)
    print("My rank is: ", rank, " and my distributed data has length: ", len(data_distributed['x']))

    # make sure we still have all particles
    n_local_distributed = len(data_distributed['x'])
    n_global_distributed = partition.comm.reduce(n_local_distributed)
    assert n_global_distributed == n_local * nranks

    if rank == 0:
        assert n_global_distributed == n_local * partition.nranks

    # validate that each particle is in local extent
    bbox = np.array([
       np.array(origin) * box_size,
       (np.array(origin) + np.array(extent)) * box_size
    ]).T
    print("For rank ", rank, " bbox is: ", bbox)
    is_valid = np.ones(n_local_distributed, dtype=np.bool_)    # AC - `after` vs. `distributed`
    for i, x in enumerate(coord_keys):
        is_valid &= data_distributed[x] >= bbox[i, 0]
        is_valid &= data_distributed[x] < bbox[i, 1]
    assert np.all(is_valid)
    
    # subvolumes should add up to the full volume (box_size^3)
    sidelengths = np.empty(ndims, dtype=np.float64)
    for dim in range(ndims):
        sidelengths[dim] = box_size * (extent[dim] - origin[dim])
    rank_subvolume = np.product(sidelengths)
    print("For rank ", rank, " subvolume is: ", rank_subvolume)
    if rank == 0:
        global_volume = partition.comm.reduce(rank_subvolume, root = rank)
        print("Global volume is: ", global_volume)
        print("Box volume is: ", box_size**ndims)
        assert global_volume == box_size**ndims
    # AC - Some sort of problem happening here??? Rank 0 seems to not be happening at all...
        
# Try feeding a box_size and nranks combo that prevents a commensurate topology? Something like below?
#test_distribute(box_size = 33) # run with np 4

@pytest.mark.mpi
def test_1d(box_size = 200):
    test_distribute(ndims = 1, box_size = box_size)
    
@pytest.mark.mpi
def test_2d(box_size = 200):
    test_distribute(ndims = 2, box_size = box_size)
    
@pytest.mark.mpi
def test_3d(box_size = 200):
    test_distribute(ndims = 3, box_size = box_size)
    
@pytest.mark.mpi
def test_4d(box_size = 200):
    test_distribute(ndims = 4, box_size = box_size)

#test_1d()

test_2d()

#test_3d()

#test_4d()

#class TestMpipartition(unittest.TestCase):
#    """Tests for `mpipartition` package."""

#    def setUp(self):
#        """Set up test fixtures, if any."""

#    def tearDown(self):
#        """Tear down test fixtures, if any."""

#    def test_000_something(self):
#        """Test something."""
