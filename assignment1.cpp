#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>

int main(int argc, char* argv[]) {
    clock_t start = clock();  // Get the start time
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000; // Number of random samples
    const int seed = 12345; // Initial seed

    // Initialize the random number generator using the leapfrog method
    long long n_prev = seed;
    const long long a = 1664525;
    const long long m = 1LL << 32;
    const long long c = 1013904223;

    // Each process generates its own set of random numbers
    long long n_local = n_prev;
    for (int i = 0; i < rank; ++i) {
        n_local = (a * n_local + c) % m;
    }

    int count_inside_circle = 0;

    // Simulate the dart-throwing process
    for (int i = 0; i < N / size; ++i) {
        double x = (double)n_local / m * 2.0 - 1.0;
        n_local = (a * n_local + c) % m;
        double y = (double)n_local / m * 2.0 - 1.0;
        n_local = (a * n_local + c) % m;

        double distance = x * x + y * y;
        if (distance <= 1.0) {
            count_inside_circle++;
        }
    }

    // Gather results from all processes
    int total_inside_circle;
    MPI_Reduce(&count_inside_circle, &total_inside_circle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate π
        double pi_estimate = 4.0 * total_inside_circle / (N * 1.0);
        std::cout << "Estimated π value: " << pi_estimate << std::endl;
    }

    MPI_Finalize();

    clock_t stop = clock();   // Get the stop time

    double duration = (static_cast<double>(stop - start) / CLOCKS_PER_SEC);

    std::cout << "Time taken by code: " << duration << " seconds" << std::endl;

    return 0;
}

// compile 
// mpic++ assignment1.cpp -o assignment1

// run
// mpirun -np 1 assignment1


//compile windows 
//g++ -I"C:\Program Files (x86)\Microsoft MPI\SDK\Include" -o assignment1.exe assignment1.cpp "C:\Program Files (x86)\Microsoft MPI\SDK\Lib\x64\msmpi.lib"

//run
//"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -np 3 assignment1.exe