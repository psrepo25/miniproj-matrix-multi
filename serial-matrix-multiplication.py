import time
import numpy as np

def matrix_multiply(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    result = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

if __name__ == "__main__":
    MATRIX_SIZE = 500  # Match with your MPI test
    np.random.seed(42)
    A = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
    B = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
    start = time.time()
    result = matrix_multiply(A, B)
    end = time.time()
    print(f"Serial runtime: {end - start:.6f} s")