from mpi4py import MPI
from time import time
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
'''
# cannon's algorithm
All the matrices and submatrices should be square matrices
A*B = C
A and B are 4*4 matrix
decompose them into 4 processors (4 sub matrices by corners)

          0   1
         R0   R1
        1 2 | 3 4       0
        5 6 | 7 8
A   =   ---------
        9 1 | 2 3       1
        4 5 | 6 7 
         R3   R4
similarly B is also divided into 4 ranks

    M0A0 M1A1         M0B0 M1B1
    M2A2 M3A3         M2B2 M3B3
    
    M0A01:
    M means the submatrix
    0 is the number of that submatrix
    A is the parent of that submatrix
    1 is the rank it is currently in 
    here 0,1,2,3 are the ranks
Step 1:
    Row Ring Shift for all A (the number of ring shift will be based on the row number,
    0 will go zero ring shift and 1 will go 1 ring shift, if there is 2 then it will go 2 ring shifts)

    Column Ring Shift for all B (similar to A, the number of ring shift will be based on the column number)
    Current matrices in the Ranks: 
    M0A0 M1A1         M0B0 M3B1
    M3A2 M2A3         M2B2 M1B3
    
    Now multiply the submatrix present in each processors locally

Step 2:
    Now do 1 ring shifts for all the rows in A and for all the columns in B
    Multiple the submatrices present and add the values to the previous results

Repeat this step two for [sqrt(processors)-1] number of times and keep adding the values

Gather all the values and that is final result

Note: Sqrt(precessors) should be and integer than only this algorithm will work
      Rows - Left Shift
      Columns - Up Shift
'''