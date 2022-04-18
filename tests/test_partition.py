#!/usr/bin/env python

"""Tests for `mpipartition` package."""

import sys
sys.path.append('MPIPartition/mpipartition')
import mpipartition
import unittest
import pytest
from partition import Partition
import numpy as np

# create a partition of the unit cube with available MPI ranks

def test_partition(ndims, verbose = True):
    partition = Partition(ndims=ndims)
    rank = partition.rank
    comm = partition.comm
    nranks = partition.nranks

    if verbose:
        if rank == 0:
            print(f"\nPartition rank: {partition.rank}")
            print(f"Number of ranks: {partition.nranks}")
            print(f"Volume decomposition: {partition.decomposition}")
            print(f"Subvolume extent: {partition.extent}")
            print(f"Subvolume origin: {partition.origin}")

        else:
            print(f"\nPartition rank: {partition.rank}")
            print(f"Subvolume extent: {partition.extent}")
            print(f"Subvolume origin: {partition.origin}")

# Here, check if there are as many ranks as expected and they add up to the full "rank_subvolume"? (Basically, each rank should have a "volume" of 1, and the total volume should be the total number of ranks)
    rank_sidelengths = np.empty(ndims, dtype=float) # AC - is this supposed to be something like float64?
    for dim in range(ndims):
        rank_sidelengths[dim] = partition.decomposition[dim] * partition.extent[dim] # + partition.origin[dim]  # AC - do I care about the origin?
    rank_subvolume = np.prod(rank_sidelengths)
    global_volume = comm.reduce(rank_subvolume)
    assert rank_subvolume == 1.0
    if rank == 0:
        assert global_volume == float(nranks)
        
# Mess around with neighbors?
# Maybe neighbors should have an origin that is equal to my origin +/- my extent? That's for get_neighbor, anyway
    # Number of neighbors + 1 should be nranks
    assert partition.neighbors_unique_count + 1 == nranks
    
    # Number of dimensions should be what I said it would be
    assert len(partition.decomposition) == ndims
    
    # Each process should have a length less than or equal to 1, and its origin should be positive, and its subvolume should be 1
    for dim in range(ndims):
        assert partition.origin[dim] >= 0.0
        assert partition.extent[dim] - partition.origin[dim] <= 1.0

# RUN THE TESTS FOR DIFFERENT DIMENSIONS
test_partition(1)
# Eventually learn how to do it this way
@pytest.mark.mpi
def test_partition_1d():
    test_partition(1)
    
@pytest.mark.mpi
def test_partition_2d():
    test_partition(2)
    
@pytest.mark.mpi
def test_partition_3d():
    test_partition(3)
    
@pytest.mark.mpi
def test_partition_4d():
    test_partition(4)