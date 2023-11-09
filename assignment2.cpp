#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>

void printArray(const std::vector<int>& arr) {
    for (const auto& num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double start_time, end_time;

    // Master process initializes the array
    if (world_rank == 0) {
        // Get the size of the array from the command line
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int array_size = std::atoi(argv[1]);
        std::vector<int> data(array_size);

        // Populate the array with random numbers
        srand(static_cast<unsigned>(time(nullptr)));
        for (int i = 0; i < array_size; ++i) {
            data[i] = rand() % 100;  // Adjust the range as needed
        }

        // Print the unsorted array
        std::cout << "Unsorted Array (Master): ";
        printArray(data);

        // Record start time
        start_time = MPI_Wtime();

        // Partition the array and send to slave processes
        int chunk_size = array_size / world_size;
        for (int i = 1; i < world_size; ++i) {
            MPI_Send(&data[i * chunk_size], chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        data.resize(chunk_size); // Resize the master's array

        // Gather sorted large buckets from slave processes
        std::vector<int> recvcounts(world_size);
        std::vector<int> displs(world_size);

        MPI_Gather(&chunk_size, 1, MPI_INT, &recvcounts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

        int total_size = 0;
        if (world_rank == 0) {
            displs[0] = 0;
            total_size = recvcounts[0];
            for (int i = 1; i < world_size; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
                total_size += recvcounts[i];
            }
        }

        std::vector<int> sorted_data(total_size);
        MPI_Gatherv(nullptr, 0, MPI_INT, &sorted_data[0], &recvcounts[0], &displs[0], MPI_INT, 0, MPI_COMM_WORLD);

        // Record end time
        end_time = MPI_Wtime();

        // Perform final sorting and output the sorted array
        std::sort(sorted_data.begin(), sorted_data.end());
        std::cout << "Sorted Array (Master): ";
        printArray(sorted_data);

        // Print execution time
        std::cout << "Execution Time: " << end_time - start_time << " seconds" << std::endl;

    } else {
        // Slave processes receive partition and perform bucket sort
        int chunk_size;
        MPI_Status status;

        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &chunk_size);

        std::vector<int> local_data(chunk_size);
        MPI_Recv(&local_data[0], chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform bucket sort on the local data
        std::sort(local_data.begin(), local_data.end());

        // Send the local sorted data to the master
        MPI_Gather(&chunk_size, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&local_data[0], chunk_size, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
