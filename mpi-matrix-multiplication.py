from mpi4py import MPI
import numpy as np

def matrix_multiply_partial(A_part, B):
    m = len(A_part)
    n = len(A_part[0])
    p = len(B[0])
    result = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A_part[i][k] * B[k][j]
    return result

# ...existing code...

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Timing: Start total timer
    total_start = MPI.Wtime()

    # Only the root process initializes the full matrices
    # ...existing code...
    if rank == 0:
        MATRIX_SIZE = 500  # Change this for different tests
        A = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
        B = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
        A_rows = len(A)
        rows_per_proc = A_rows // size
        extras = A_rows % size
        counts = [rows_per_proc + (1 if i < extras else 0) for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]
# ...existing code...
    else:
        B = None
        counts = None
        displs = None

    # Timing: Data distribution start
    dist_start = MPI.Wtime()

    # Broadcast B to all processes
    B = comm.bcast(B if rank == 0 else None, root=0)

    # Scatter rows of A
    counts = comm.bcast(counts if rank == 0 else None, root=0)
    displs = comm.bcast(displs if rank == 0 else None, root=0)
    local_rows = counts[rank]
    local_A = comm.scatter([A[displs[i]:displs[i]+counts[i]] if rank == 0 else None for i in range(size)], root=0)

    # Timing: Data distribution end, computation start
    dist_end = MPI.Wtime()
    comp_start = dist_end

    # Each process computes its part
    local_result = matrix_multiply_partial(local_A, B)

    # Timing: Computation end, gathering start
    comp_end = MPI.Wtime()
    gather_start = comp_end

    # Gather results at root
    gathered = comm.gather(local_result, root=0)

    # Timing: Gathering end
    gather_end = MPI.Wtime()

    # Print local timings
    print(f"Rank {rank}: Data distribution time: {dist_end - dist_start:.6f} s, Computation time: {comp_end - comp_start:.6f} s, Gathering time: {gather_end - gather_start:.6f} s")

    if rank == 0:
        # Flatten the gathered results
        result = [row for part in gathered for row in part]
        print("Distributed Result:")
        #for row in result:
            #print(len(row))
        total_end = MPI.Wtime()
        print(f"Total parallel runtime: {total_end - total_start:.6f} s")
        # To compute speedup, run your serial code and record its time as time_serial
        # time_serial = ... (set this manually after running serial version)
        # speedup = time_serial / (total_end - total_start)
        # print(f"Speedup: {speedup:.2f}")