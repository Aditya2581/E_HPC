# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
#
# dim = (4, 4)
# period = (True, True)
# reorder = False
#
# crt2d = comm.Create_cart(dim, period, reorder)
# local_row, local_col = crt2d.Get_coords(rank)
# # print(f"rank: {rank} ({local_row}, {local_col})")
# # print(crt2d.Get_cart_rank((0, 1)))
# dir = 1
# disp = -1
#
# # left, right = crt2d.Shift(dir, disp)
# # print(rank, left, right)
# # print(crt2d.Get_coords(left), crt2d.Get_coords(right))
# sendb = rank*2
# recvb, status = crt2d.Shift(dir, disp)
# print(f'rank: {rank}, sendbuf: {sendb}, recvbuf: {recvb}, "status": {status}')
#
# # if rank == 9:
# #    left, right = crt2d.Shift(dir, disp)
# #    print(left, right)
# #    print(crt2d.Get_coords(left), crt2d.Get_coords(right))
