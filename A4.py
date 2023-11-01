from mpi4py import MPI
from time import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# serial sort

def eo_sort(arr, arr_len):
    s = 0
    for j in range(arr_len):
        for i in range((arr_len // 2) - s):
            if arr[s + i * 2 + 1] < arr[s + i * 2]:
                k = arr[s + i * 2]
                arr[s + i * 2] = arr[s + i * 2 + 1]
                arr[s + i * 2 + 1] = k
        if s == 0:
            s = 1
        else:
            s = 0


start_time = time()
ran = 100
N = 16
# array = np.random.randint(ran, size=N)
# eo_sort(arr, np.size(arr))
# end_time = time()
# print(end_time - start_time)
# print(array)

# if rank != size - 1:
#     N = int(N / size)
#     array = np.sort(np.random.randint(ran, size=N))
#     print(rank, array)
# else:
#     N = N - (size - 1) * int(N / size)
#     array = np.sort(np.random.randint(ran, size=N))
#     print(rank, array)

if rank==0:
    array = [28,64,67,90]
elif rank==1:
    array = [2,14,43,76]
elif rank==2:
    array = [22,47,60,78]
else:
    array = [18,65,85,87]
print(rank,array)
#initial = comm.gather(array, root=0)
#if rank == 0:
#    print(np.array(initial))
#print("anything")



s = 0
if s==0:
    if (rank+s)%2==0:
        r = comm.sendrecv(array,dest=rank+1, sendtag=0, source=rank+1, recvtag=0)


for j in range(size):
    i = 0
    for k in range(2 * ((size // 2) - s)):
        if k % 2 == 0:
            r = comm.sendrecv(array, dest=s + i * 2 + 1, sendtag=0, source=s + i * 2 + 1, recvtag=0)
            print("sendrecv together j={} k={} i={} rank={} array={} r={}".format(j,k,i,rank,array,r))
            array = np.sort(np.concatenate((array, r)))[:N]
            i += 1
        else:
            array = np.sort(np.concatenate((array, r)))[N:]

    if s == 0:
        s = 1
    else:
        s = 0
#
#
#
#
#
#
#
#
#
#
# if size % 2 == 1:
#    # skip 1st in odd run
#    # skip last in even run
#    skip = 1
# else:
#    # skip 1st and last in odd run
#    # skip none in even run
#    skip = 0
# for j in range(size):
#    if skip == 0:
#        if s == 0:
#            ignore = []
#        else:
#            ignore = [0, size - 1]
#    else:
#        if s == 0:
#            ignore = [size - 1]
#        else:
#            ignore = [0]
#
#    for i in range(size):
#        if i not in ignore:
#            r = np.zeros(N)
#            eo_sort(array, np.size(array))
#            comm.send(array, dest=s + i * 2 + 1)
#            # r = comm.recv(source=s+i*2)
#            # comm.sendrecv(array, dest=s + i * 2 + 1, sendtag=0, recvbuf=N*4, source=s + i * 2 + 1, recvtag=0)
#            if s == 0:
#                if i % 2 == 0:
#                    comm.sendrecv(array, dest=i + 1, sendtag=0, recvbuf=N * 4, source=i + 1, recvtag=0)
#                    array = np.sort(np.concatenate(array, r))[:N]
#                else:
#                    array = np.sort(np.concatenate(array, r))[N:]
#            else:
#                if i % 2 == 0:
#                    array = np.sort(np.concatenate(array, r))[N:]
#                else:
#                    array = np.sort(np.concatenate(array, r))[:N]
#    if s == 0:
#        s = 1
#    else:
#        s = 0
final = comm.gather(array, root=0)
print(final)
if rank == 0:
    print(final)
