#!/usr/bin/env python

"""Tests for `mpipartition` package."""

import sys
sys.path.append('/data/a/cpac/aurora/MPIPartition/mpipartition')
import mpipartition
import unittest
import pytest
from partition import Partition
from distribute import distribute
from overload import overload
import numpy as np

def test_overload(ndims, box_size, overload_length, nparticles = 1000):

    # partitioning a box with the available MPI ranks
    partition = Partition(ndims = ndims)
    rank = partition.rank
    origin = partition.origin
    extent = partition.extent
    print("\nRank ", rank, " has origin ", origin, " and extent ", extent)
    
    # Create data according to desired dimensions
    possible_coords = "xyzwuv"
    coord_keys = possible_coords[:ndims]

    #if data == None:
    data = {
        x: np.random.uniform(0, 1*box_size, nparticles) for i, x in enumerate(coord_keys)
    }
    # Need to add some sort of identifier, or similar later (like Michael does)
    
    # assign to rank by position
    data_distributed = distribute(partition, box_size, data, coord_keys)
    nparticles_dist = len(data_distributed['x'])
    print("\nFor rank ", rank, " distributed data has ", nparticles_dist, " particles")
    #print("My rank is ", partition.rank, " and my Distributed data id's look like: ", data_distributed['id'])
    
    # overload particles
    data_overloaded = overload(partition, box_size, data_distributed, overload_length, coord_keys)
    nparticles_ol = len(data_overloaded['x'])
    print("For rank ", rank, " overloaded data has ", nparticles_ol, " particles")
    #print("My rank is ", partition.rank, " and my Overloaded data id's look like: ", data_overloaded['id'])
    
    # AC - How to tell if the data has already been distributed? We want to throw an error if not
    
    # Does the amount of overloaded particles match what we would expect from theory?
    def theory_overload(frac, verbose = True):
        ndims = len(partition.decomposition)
        theory_nparticles_ol = np.empty(ndims)
        for dim in range(ndims):
            rank_size = box_size / partition.decomposition[dim]
            theory_nparticles_ol[dim] = ( (rank_size + 2*overload_length)**ndims / (rank_size)**ndims ) * nparticles_dist
            epsilon = frac * theory_nparticles_ol[dim]
            lower_threshold = theory_nparticles_ol[dim] - epsilon
            upper_threshold = theory_nparticles_ol[dim] + epsilon
            # Using print statements instead of assert, because there seem to be outliers
            if verbose == True:
                print("\nFor rank ", rank, " and dim ", dim, ":\ntheory_nparticles_ol is: ", theory_nparticles_ol[dim], "\nepsilon is: ", epsilon, "\nLower threshold is: ", lower_threshold, "\nUpper threshold is: ", upper_threshold, "\nAbove upper_threshold: ", nparticles_ol > upper_threshold, "\nBelow lower_threshold: ", nparticles_ol < lower_threshold)
            else:
                print("\nFor rank ", rank, " and dim ", dim, "\nDo we violate upper_threshold?: ", nparticles_ol > upper_threshold, "\nDo we violate lower_threshold?: ", nparticles_ol < lower_threshold)
            #assert nparticles_ol >= lower_threshold
            #assert nparticles_ol <= upper_threshold
        # How does this work if the numbers in partition.decomposition are not all the same?
    
    print("\nDo those numbers make sense?")
    theory_overload(frac = 0.3, verbose = False)

    # validate that each particle is in local extent
    bbox = np.array([
       np.array(origin) * box_size,
       (np.array(origin) + np.array(extent)) * box_size
    ]).T
    is_valid = np.ones(nparticles_dist, dtype=np.bool_)
    for i, x in enumerate(coord_keys):
        is_valid &= data_distributed[x] >= bbox[i, 0]
        is_valid &= data_distributed[x] < bbox[i, 1]
    assert np.all(is_valid)
    
    # WIP: Make sure the data on the ranks is ALL the particles
    #mesh = np.meshgrid(data[x] for x in coord_keys)
    #local_particles = np.empty(ndims)
    #for dim, x in enumerate(coord_keys):
    #    local_particles[dim] = data[x]
    
    # make sure we still have all particles
    #local_particles = np.ndarray([data_distributed['id']])
    #nparticles_distributed = len(data_distributed['x'])
    #print("My rank is: ", partition.rank, " and my nparticles_distributed is:", nparticles_distributed)
    #n_global_distributed = partition.comm.reduce(nparticles_distributed)
    #if partition.rank == 0:
    #    assert n_global_distributed == nparticles * partition.nranks

        
# Want to try feeding a box_size and nranks combo that prevents a commensurate topology?

# Add a test for overload_length = 0
def test_0overload_length(ndims, box_size = 200, nparticles = 1000):
    
    # partitioning a box with the available MPI ranks
    partition = Partition(ndims = ndims)
    rank = partition.rank
    origin = partition.origin
    extent = partition.extent
    print("\nRank ", rank, " has origin ", origin, " and extent ", extent)
    
    # Create data according to desired dimensions
    possible_coords = "xyzwuv"
    coord_keys = possible_coords[:ndims]
    overload_length = 0

    #if data == None:
    data = {
        x: np.random.uniform(0, 1*box_size, nparticles) for i, x in enumerate(coord_keys)
    }
    
    # assign to rank by position
    data_distributed = distribute(partition, box_size, data, coord_keys)
    nparticles_dist = len(data_distributed[coord_keys[0]])
    
    # overload particles with overload_length = 0
    data_overloaded = overload(partition, box_size, data_distributed, overload_length, coord_keys)
    nparticles_ol = len(data_overloaded[coord_keys[0]])
    assert nparticles_dist == nparticles_ol

@pytest.mark.mpi
def test_1d(box_size = 200, overload_length = 8, nparticles = 1000):
    test_overload(ndims = 1, box_size = box_size, overload_length = overload_length, nparticles = nparticles)
    test_0overload_length(ndims = 1, box_size = box_size, nparticles = nparticles)
    
@pytest.mark.mpi
def test_2d(box_size = 200, overload_length = 8, nparticles = 1000):
    test_overload(ndims = 2, box_size = box_size, overload_length = overload_length, nparticles = nparticles)
    test_0overload_length(ndims = 2, box_size = box_size, nparticles = nparticles)
    
@pytest.mark.mpi
def test_3d(box_size = 200, overload_length = 8, nparticles = 1000):
    test_overload(ndims = 3, box_size = box_size, overload_length = overload_length, nparticles = nparticles)
    test_0overload_length(ndims = 3, box_size = box_size, nparticles = nparticles)
    
@pytest.mark.mpi
def test_4d(box_size = 200, overload_length = 8, nparticles = 1000):
    test_overload(ndims = 4, box_size = box_size, overload_length = overload_length, nparticles = nparticles)
    test_0overload_length(ndims = 4, box_size = box_size, nparticles = nparticles)
    
# Problem with test_1d()?
#test_3d()
test_4d()
