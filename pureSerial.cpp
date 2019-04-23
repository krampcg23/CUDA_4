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

void fillQueue(vertex* V, int* E, int n, std::queue<int> &q, bool* visited) {
    visited[n] = true;
    int start = V[n].start;
    int length = V[n].numAdj;
    if (length == 0) return;
    for (int i = start; i < start + length; i++) {
        if(!visited[E[i]]){
            q.push(E[i]);
        }
    }
}

void runBFS(vertex* V, int* E, int vertices, int edges, bool* visited) {
    for (int i = 0; i < vertices; i++) {
        visited[i] = false;
    }

    std::queue<int> q;
    fillQueue(V, E, 1, q, visited);
    
    while(!q.empty()) {
        int vert = q.front();
        q.pop();
        if (!visited[vert]) {
            fillQueue(V, E, vert, q, visited);
        }
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Incorrect Usage, please use ./main [filename] " << std::endl;
    }

    clock_t begin = clock();
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
    runBFS(V, E, vertices, edges, visited);
    clock_t end = clock();
    double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
    std::cout << "Total Sequential Execution Time: " << timeSec << std::endl;

    return 0;

}
