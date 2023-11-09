#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <chrono>

// Linear Congruential Generator parameters
const unsigned long long a = 1664525ULL;
const unsigned long long c = 1013904223ULL;
const unsigned long long m = 4294967296ULL;  // 2^32

// Function to perform modulo exponentiation
unsigned long long mod_pow(unsigned long long base, unsigned long long exponent, unsigned long long mod) {
    unsigned long long result = 1;
    for (unsigned long long i = 0; i < exponent; ++i) {
        result = (result * base) % mod;
    }
    return result;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int num_samples = 1000000;  // Number of random samples per process
    const int num_total_samples = num_samples * num_procs;

    // Calculate jump constants A and C
    unsigned long long k = num_procs;
    unsigned long long A = mod_pow(a, k, m);
    unsigned long long C = 0;
    for (unsigned long long i = 0; i < k; ++i) {
        C += mod_pow(a, i, m);
    }
    C %= m;

    // Leapfrog method: Seed the random number generator only on the master process
    unsigned long long seed;
    if (rank == 0) {
        seed = time(NULL);
    }

    // Broadcast seed from master to all other processes
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Calculate the starting point for the leapfrog method
    unsigned long long leap_start = seed;
    for (int i = 1; i <= rank; ++i) {
        leap_start = (A * leap_start + C) % m;
    }

    // Measure parallel execution time
    auto par_start_time = std::chrono::high_resolution_clock::now();
    int count_inside_circle = 0;
    unsigned long long n = leap_start;
    for (int i = 0; i < num_samples; ++i) {
        double x = static_cast<double>(n) / m * 2.0 - 1.0;
        n = (A * n + C) % m;
        double y = static_cast<double>(n) / m * 2.0 - 1.0;
        n = (A * n + C) % m;
        if (x * x + y * y <= 1.0) {
            count_inside_circle++;
        }
    }

    // Sum up counts from all processes
    int total_inside_circle;
    MPI_Reduce(&count_inside_circle, &total_inside_circle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto par_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> par_elapsed_time = par_end_time - par_start_time;

    // Calculate speedup
    double speedup = par_elapsed_time.count();

    if (rank == 0) {
        // Calculate the estimate of π
        double pi_estimate = 4.0 * total_inside_circle / static_cast<double>(num_total_samples);
        std::cout << "Estimated value of π: " << pi_estimate << std::endl;
        std::cout << "Parallel Execution Time: " << par_elapsed_time.count() << " seconds" << std::endl;
        std::cout << "Speedup: " << speedup << std::endl;
    }

    MPI_Finalize();

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