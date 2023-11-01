from mpi4py import MPI
import random
from time import time
import matplotlib.pyplot as plt
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


'''
Newtype.commit()    # to allocate memory
Newtype.free()      # to free the memory

MPI.TYPE.Create_vector(count, blocklenngth, stride)
count = how many sets of data u wnat
blocklength = how many data is in one set
stride = distance between starting elements of two sets 
'''

'''
1

if rank == 0:
    a = np.arange(6,dtype=np.float64).reshape((3, 2))
    #a = np.array([1.2, 2.3, 3.4, 4.5, 5.798465, 6, 7, 8, 9], dtype=np.float64)
    #print(type(a))
else:
    a = None

count = 3  # no of blocks
blocklength = 1  # no of elements in each block
stride = 2
vect = MPI.DOUBLE.Create_vector(count, blocklength, stride)
vect.Commit()

if rank == 0:
    comm.Send([(np.frombuffer(a.data, np.float64, offset=8)), 1, vect], dest=1)
else:
    b = np.empty(count * blocklength, dtype=np.float64)
    comm.Recv(b, source=0)
    print(b)
vect.Free()
'''

'''
2

if rank == 0:
    a = np.arange(20, dtype=np.int32)
    #a = np.arange(6,dtype=np.float64).reshape((3, 2))
    #a = np.array([1.2, 2.3, 3.4, 4.5, 5.798465, 6, 7, 8, 9], dtype=np.float64)
    print(a)
else:
    a = None

blocklength = 2  # no of elements in each block
displs = [0,5,8,13,18]
vect = MPI.INT.Create_indexed_block(blocklength, displs)
vect.Commit()

if rank == 0:
    comm.Send([(np.frombuffer(a.data, np.int32, offset=0)), 1, vect], dest=1)
else:
    b = np.empty((len(displs)) * blocklength, dtype=np.int32)
    comm.Recv(b, source=0)
    print(b)
vect.Free()
'''

'''
3
if rank == 0:
    a = np.arange(16,dtype=np.int32).reshape((4, 4))
    # a = np.array([1.2, 2.3, 3.4, 4.5, 5.798465, 6, 7, 8, 9], dtype=np.float64)
    print(a)
else:
    a = None

size = (4,4)
subsize = (2,2)
start = (2,2)
order = MPI.ORDER_C

vect = MPI.INT.Create_subarray(size, subsize, start, order=order)
vect.Commit()

if rank == 0:
    print("rank: ", rank)
    print(type(vect))
    print("size taken in rank 0: ", sys.getsizeof(vect))
    comm.Send([(np.frombuffer(a.data, np.int32, offset=00)), 1, vect], dest=1)
else:
    print("rank: ", rank)
    print(type(vect))
    print("size wasted: ", sys.getsizeof(vect))
    b = np.empty(subsize, dtype=np.int32)
    comm.Recv(b, source=0)
    print(b)
vect.Free()
'''

##############
# ring shift matrix vector multiply
# [a1 a2 a3 a4   [1       [2        [3
#  b1 b2 b3 b4    2 ---->  3  ---->  4
#  c1 c2 c3 c4    3  ring  4         1
#  d1 d2 d3 d4]   4] shift 1]        2]
#
