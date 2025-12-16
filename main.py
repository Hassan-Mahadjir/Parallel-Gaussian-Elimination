from mpi4py import MPI
import numpy as np
import sys

# ----------------------------------
# MPI ENVIRONMENT - create communicator
# ----------------------------------
class MPIEnvironment:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.p = self.comm.Get_size()


# ----------------------------------
# PARALLEL GAUSSIAN ELIMINATION
# ----------------------------------
class ParallelGaussianElimination:
    def __init__(self, mpi_env):
        self.comm = mpi_env.comm
        self.rank = mpi_env.rank
        self.p = mpi_env.p

        self.n = None
        self.augmented_matrix = None
        self.original_matrix = None
        self.x = None

        self.forward_time = 0.0
        self.backward_time = 0.0
        self.total_time = 0.0

    # ----------------------------------
    # MATRIX INITIALIZATION
    # ----------------------------------
    def initialize_matrix(self):
        try:
            if self.rank == 0:
                if len(sys.argv) < 3:
                    raise RuntimeError(
                        "Usage:\n"
                        "  random n\n"
                        "  file matrix.txt"
                    )

                mode = sys.argv[1]

                if mode == "random":
                    self.n = int(sys.argv[2])
                    self.augmented_matrix = np.random.rand(self.n, self.n + 1) * 10

                elif mode == "file":
                    filename = sys.argv[2]
                    with open(filename, "r") as f:
                        self.n = int(f.readline())
                        self.augmented_matrix = np.zeros((self.n, self.n + 1))
                        for i in range(self.n):
                            self.augmented_matrix[i] = list(
                                map(float, f.readline().split())
                            )

                else:
                    raise RuntimeError("Invalid mode. Use 'random' or 'file'.")

                self.original_matrix = self.augmented_matrix.copy()

            else:
                self.n = None
                self.augmented_matrix = None

            # Broadcast
            self.n = self.comm.bcast(self.n, root=0)
            self.augmented_matrix = self.comm.bcast(self.augmented_matrix, root=0)

        except Exception as e:
            if self.rank == 0:
                print("ERROR:", e)
            self.comm.Abort(1)
        
    # ----------------------------------
    # PARALLEL FORWARD ELIMINATION
    # ----------------------------------
    def parallel_forward_elimination(self):
        start_time = MPI.Wtime()

        for k in range(self.n):
            pivot_owner = k % self.p

            if self.rank == pivot_owner:
                pivot_value = self.augmented_matrix[k][k]
                self.augmented_matrix[k] /= pivot_value

            self.comm.Bcast(self.augmented_matrix[k], root=pivot_owner)

            for i in range(k + 1, self.n):
                if i % self.p == self.rank:
                    factor = self.augmented_matrix[i][k]
                    self.augmented_matrix[i] -= factor * self.augmented_matrix[k]

            self.comm.Barrier()

        end_time = MPI.Wtime()
        self.forward_time = end_time - start_time

    # ----------------------------------
    # PARALLEL BACK SUBSTITUTION
    # ----------------------------------
    def parallel_back_substitution(self):
        start_time = MPI.Wtime()

        self.x = np.zeros(self.n)

        for i in range(self.n - 1, -1, -1):
            pivot_owner = i % self.p

            if self.rank == pivot_owner:
                self.x[i] = (
                    self.augmented_matrix[i][-1]
                    / self.augmented_matrix[i][i]
                )

            self.x[i] = self.comm.bcast(self.x[i], root=pivot_owner)

            for r in range(i):
                if r % self.p == self.rank:
                    self.augmented_matrix[r][-1] -= (
                        self.augmented_matrix[r][i] * self.x[i]
                    )

            self.comm.Barrier()

        end_time = MPI.Wtime()
        self.backward_time = end_time - start_time

    # ----------------------------------
    # PERFORMANCE REPORT
    # ----------------------------------
    def report_performance(self):
        self.total_time = self.forward_time + self.backward_time

        max_forward = self.comm.reduce(self.forward_time, op=MPI.MAX, root=0)
        max_backward = self.comm.reduce(self.backward_time, op=MPI.MAX, root=0)
        max_total = self.comm.reduce(self.total_time, op=MPI.MAX, root=0)

        if self.rank == 0:
            print("\n--- Performance Results ---")
            print(f"Matrix size (n): {self.n}")
            print(f"Number of processes (p): {self.p}")
            print(f"Forward elimination time: {max_forward:.6f} s")
            print(f"Back substitution time: {max_backward:.6f} s")
            print(f"Total execution time T_p: {max_total:.6f} s")

    # ----------------------------------
    # DISPLAY RESULTS
    # ----------------------------------
    def display_results(self):
        if self.rank == 0:
            print("\nOriginal augmented matrix [A | b]:")
            print(self.original_matrix)

            print("\nUpper triangular matrix U:")
            print(self.augmented_matrix)

            print("\nSolution vector x:")
            print(self.x)


# ----------------------------------
# MAIN PROGRAM
# ----------------------------------
def main():
    mpi_env = MPIEnvironment()
    solver = ParallelGaussianElimination(mpi_env)

    global_start = MPI.Wtime()
    processor_name = MPI.Get_processor_name()
    print(f"Rank {mpi_env.rank}/{mpi_env.p} running on {processor_name}", flush=True)

    solver.initialize_matrix()
    solver.parallel_forward_elimination()
    solver.parallel_back_substitution()

    global_end = MPI.Wtime()

    solver.report_performance()
    solver.display_results()

    if mpi_env.rank == 0:
        print(f"\nOverall wall-clock time: {global_end - global_start:.6f} s")


if __name__ == "__main__":
    main()
