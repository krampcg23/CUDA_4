#include <iostream>
#include <fstream>
#include <queue>
#include <sstream>
#include <string>
#include <ctime>
#include <assert.h>

struct vertex {
    int start;
    int numAdj;
    vertex() { numAdj = 0; start = -1; }
};


__global__ void parallelBFS(vertex* V, int* E, bool* q, bool* visited, int* cost, int vertices, bool* flags) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //int numThreads = blockDim.x * gridDim.x;
    int j = 0;

    // Uncomment for load balance attempt
 //   for (int j = 0; j < vertices; j += numThreads) {
        int id = tid + j; 
        //if (id > vertices) continue;
        if (id > vertices) return;
        
        if (q[id] == true) {
            q[id] = false;
            int start = V[id].start;
            int length = V[id].numAdj;
            if (length == 0) return;
            for (int i = start; i < start + length; i++) {
                int adjacent = E[i];
                if (visited[adjacent] == false) {
                    cost[adjacent] = min(cost[adjacent], cost[id] + 1);
                    flags[adjacent] = true;
                }
            }
        }       
  //  }
    return;
}

__global__ void parallelBFS_flags(vertex* V, int* E, bool* q, bool* visited, bool* qNotEmpty, int vertices, bool* flags) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //int numThreads = blockDim.x * gridDim.x;
    int j = 0;
    //for (int j = 0; j < vertices; j += numThreads) {
        int id = tid + j; 
        //if (id > vertices) continue;
        if (id > vertices) return;
        
        if (flags[id] == true) {
            q[id] = true;
            visited[id] = true;
            *qNotEmpty = true;
            flags[id] = false;
        }
  //  }
    return;
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Incorrect Usage, please use ./main [filename] " << std::endl;
    }
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::string filename = argv[1];
    std::ifstream file(filename);
    std::string firstLine;
    getline(file, firstLine);
    std::stringstream ss(firstLine);

    int vertices, edges, numThreads;
    ss >> numThreads;
    getline(file, firstLine);
    std::stringstream ss1(firstLine);
    ss1 >> vertices >> edges;
    vertices++; edges++;
    vertex* V = new vertex[vertices];
    int* E = new int[edges];
    E[0] = 0;

    int currentVertex = 1;
    int counter = 1;
    V[1].start = 1;
    for (int i = 0; i < edges-1; i++) {
        std::string line;
        getline(file, line);
        std::stringstream ss2(line);
        int to, from, weight;
        ss2 >> to >> from >> weight;
        if (from != currentVertex) {
            currentVertex = from;
            V[from].start = counter;
        }
        V[from].numAdj++;
        E[counter] = to;
        counter++;
    }

    bool* qNotEmpty = new bool;
    *qNotEmpty = true;

    bool* q = new bool[vertices];
    bool* visitedParallel = new bool[vertices];
    int* costParallel = new int[vertices];
    for (int i = 0; i < vertices; i++) {
        q[i] = false;
        visitedParallel[i] = false;
        costParallel[i] = 999;
    }
    q[1] = true;
    costParallel[1] = 0;

    vertex* deviceVertex;
    int* deviceEdges;
    bool* deviceQueue;
    bool* deviceVisited;
    bool* deviceQNotEmpty;
    int* deviceCost;
    bool* deviceFlags;

    cudaMalloc(&deviceVertex, sizeof(vertex) * vertices);
    cudaMalloc(&deviceEdges, sizeof(int) * edges);
    cudaMalloc(&deviceQueue, sizeof(bool) * vertices);
    cudaMalloc(&deviceVisited, sizeof(bool) * vertices);
    cudaMalloc(&deviceQNotEmpty, sizeof(bool));
    cudaMalloc(&deviceCost, sizeof(int) * vertices);
    cudaMalloc(&deviceFlags, sizeof(bool) * vertices);

    cudaMemcpy(deviceVertex, V, sizeof(vertex) * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdges, E, sizeof(int) * edges, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQueue, q, sizeof(bool) * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVisited, visitedParallel, sizeof(bool) * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQNotEmpty, qNotEmpty, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCost, costParallel, sizeof(int) * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFlags, visitedParallel, sizeof(bool) * vertices, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024, 1, 1);
    dim3 numBlocks(vertices / 1024 + 1, 1, 1);

 
    while(*qNotEmpty) {
        *qNotEmpty = false;
        cudaMemcpy(deviceQNotEmpty, qNotEmpty, sizeof(bool), cudaMemcpyHostToDevice);
        parallelBFS <<<numBlocks, threadsPerBlock>>> (deviceVertex, deviceEdges, deviceQueue, deviceVisited, deviceCost, vertices, deviceFlags);
        parallelBFS_flags <<<numBlocks, threadsPerBlock>>> (deviceVertex, deviceEdges, deviceQueue, deviceVisited, deviceQNotEmpty, vertices, deviceFlags);
        cudaThreadSynchronize();
        cudaMemcpy(qNotEmpty, deviceQNotEmpty, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time *= 0.001;
    printf("Total Execution Time:  %3.5f s \n", time);

    return 0;

}
