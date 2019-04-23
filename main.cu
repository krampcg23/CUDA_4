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
    int numThreads = blockDim.x * gridDim.x;

    for (int j = 0; j < vertices; j += numThreads) {
        int id = tid + j; 
        if (id > vertices) continue;
        
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
    }
    return;
}

__global__ void parallelBFS_flags(vertex* V, int* E, bool* q, bool* visited, bool* qNotEmpty, int vertices, bool* flags) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int numThreads = blockDim.x * gridDim.x;
    for (int j = 0; j < vertices; j += numThreads) {
        int id = tid + j; 
        if (id > vertices) continue;
        
        if (flags[id] == true) {
            q[id] = true;
            visited[id] = true;
            *qNotEmpty = true;
            flags[id] = false;
        }
    }
    return;
}

void fillQueue(vertex* V, int* E, int n, std::queue<int> &q, bool* visited, int* cost) {
    visited[n] = true;
    int start = V[n].start;
    int length = V[n].numAdj;
    if (length == 0) return;
    for (int i = start; i < start + length; i++) {
        if(visited[E[i]] == false){
            cost[E[i]] = min(cost[E[i]], cost[n] + 1);
            q.push(E[i]);
            visited[E[i]] = true;
        }
    }
}

void runBFS(vertex* V, int* E, int vertices, int edges, bool* visited, int* cost) {
    for (int i = 0; i < vertices; i++) {
        visited[i] = false;
    }

    std::queue<int> q;
    q.push(1);
    
    while(!q.empty()) {
        int vert = q.front();
        q.pop();
        fillQueue(V, E, vert, q, visited, cost);
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Incorrect Usage, please use ./main [filename] " << std::endl;
    }
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
    bool* visited = new bool[vertices];
    int* cost = new int[vertices];
    for (int i = 0; i < vertices; i++) {
        cost[i] = 999;
    }
    cost[1] = 0;
    clock_t begin = clock();
    runBFS(V, E, vertices, edges, visited, cost);
    clock_t end = clock();
    double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
    std::cout << "Sequential Execution Time: " << timeSec << std::endl;

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

    dim3 threadsPerBlock(numThreads, 1, 1);
    dim3 numBlocks(1, 1, 1);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
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
    printf("Kernel Execution Time:  %3.5f s \n", time);
    std::cout << "Speedup of: " << timeSec / time << std::endl;

    cudaMemcpy(costParallel, deviceCost, sizeof(int) * vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(visitedParallel, deviceVisited, sizeof(bool) * vertices, cudaMemcpyDeviceToHost);
    for (int i = 1; i < vertices; i++) {
        assert(visitedParallel[i] == visited[i]);
    }
    for (int i = 1; i < vertices; i++) {
        //printf("%i, %i, %i\n", i, costParallel[i], cost[i]);
        assert(costParallel[i] == cost[i]);
    }
    std::cout << "Output matches serial execution" << std::endl;

    return 0;

}
