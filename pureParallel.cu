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


__global__ void parallelBFS(vertex* V, int* E, bool* q, bool* visited, bool* qNotEmpty) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (q[id] == true && visited[id] == false) {
        q[id] = false;
        visited[id] = true;
        __syncthreads();
        int start = V[id].start;
        int length = V[id].numAdj;
        if (length == 0) return;
        for (int i = start; i < start + length; i++) {
            int adjacent = E[i];
            if (visited[adjacent] == false) {
                q[adjacent] = true;
                __syncthreads();
                *qNotEmpty = true;
            }
        }
    }
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
    for (int i = 0; i < vertices; i++) {
        q[i] = false;
        visitedParallel[i] = false;
    }
    q[1] = true;

    vertex* deviceVertex;
    int* deviceEdges;
    bool* deviceQueue;
    bool* deviceVisited;
    bool* deviceQNotEmpty;

    cudaMalloc(&deviceVertex, sizeof(vertex) * vertices);
    cudaMalloc(&deviceEdges, sizeof(int) * edges);
    cudaMalloc(&deviceQueue, sizeof(bool) * vertices);
    cudaMalloc(&deviceVisited, sizeof(bool) * vertices);
    cudaMalloc(&deviceQNotEmpty, sizeof(bool));

    cudaMemcpy(deviceVertex, V, sizeof(vertex) * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdges, E, sizeof(int) * edges, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQueue, q, sizeof(bool) * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVisited, visitedParallel, sizeof(bool) * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQNotEmpty, qNotEmpty, sizeof(bool), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(numThreads, 1, 1);
    dim3 numBlocks(vertices / numThreads + 1, 1, 1);

    while(*qNotEmpty) {
        *qNotEmpty = false;
        cudaMemcpy(deviceQNotEmpty, qNotEmpty, sizeof(bool), cudaMemcpyHostToDevice);
        parallelBFS <<<numBlocks, threadsPerBlock>>> (deviceVertex, deviceEdges, deviceQueue, deviceVisited, deviceQNotEmpty);
        cudaMemcpy(qNotEmpty, deviceQNotEmpty, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time *= 0.001;
    printf("Total Execution Time:  %3.5f s \n", time);

    return 0;

}
