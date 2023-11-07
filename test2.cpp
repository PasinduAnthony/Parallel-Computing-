#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>

// Function to perform the bucket sort
void bucketSort(std::vector<int>& local_data, int local_size, int world_size, int my_rank) {
    // Define the range of values and the number of buckets
    const int min_value = 0;
    const int max_value = 100; // Adjust this based on your requirements
    const int num_buckets = world_size;

    // Calculate the range for each bucket
    int range_per_bucket = (max_value - min_value + 1) / num_buckets;

    // Initialize local buckets
    std::vector<std::vector<int>> local_buckets(num_buckets);

    // Place data into local buckets
    for (int i = 0; i < local_size; i++) {
        int value = local_data[i];
        int bucket_index = (value - min_value) / range_per_bucket;
        local_buckets[bucket_index].push_back(value);
    }

    // Gather all local buckets to the master process
    std::vector<std::vector<int>> all_buckets(num_buckets);

    // Corrected MPI_Gather calls
    for (int i = 0; i < num_buckets; i++) {
        // Send data using local_buckets[i].data() and receive it in all_buckets[i].data()
        MPI_Gather(local_buckets[i].data(), local_buckets[i].size(), MPI_INT,
                   all_buckets[i].data(), local_buckets[i].size(), MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Sort the buckets on the master process
    if (my_rank == 0) {
        for (int i = 0; i < num_buckets; i++) {
            std::sort(all_buckets[i].begin(), all_buckets[i].end());
        }
    }

    // Broadcast the sorted buckets to all processes
    for (int i = 0; i < num_buckets; i++) {
        // Broadcast each bucket separately
        MPI_Bcast(all_buckets[i].data(), all_buckets[i].size(), MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Reconstruct the sorted data
    local_data.clear();
    for (int i = 0; i < num_buckets; i++) {
        local_data.insert(local_data.end(), all_buckets[i].begin(), all_buckets[i].end());
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Define the size of the array
    const int array_size = 100; // Adjust this based on your requirements
    int local_size = array_size / world_size;

    // Allocate memory for the local array
    std::vector<int> local_data(local_size);

    // Generate random numbers for the local array
    std::srand(static_cast<unsigned>(std::time(nullptr) + my_rank));
    for (int i = 0; i < local_size; i++) {
        local_data[i] = std::rand() % 100; // Adjust the range as needed
    }

    // Perform bucket sort
    bucketSort(local_data, local_size, world_size, my_rank);

    // Print the sorted data on each process
    for (int i = 0; i < world_size; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == i) {
            std::cout << "Process " << my_rank << ": ";
            for (int value : local_data) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
