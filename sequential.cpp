#include <iostream>
#include <fstream>
#include <queue>
#include <sstream>
#include <string>

void fillWithNull(int* arr, int num) {
    for (int i = 0; i < num; i++) {
        arr[i] = -1;
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

    int vertices, edges;
    ss >> vertices >> edges;
    int* V = new int[vertices];
    int* E = new int[edges];

    fillWithNull(V, vertices);
    fillWithNull(E, edges);

    int currentVertex = 0;
    int counter = 0;
    V[0] = 0;
    while (!file.eof()) {
        std::string line;
        getline(file, line);
        std::stringstream ss2(line);
        int to, from;
        ss2 >> to >> from;
        if (from != currentVertex) {
            currentVertex = from;
            V[from] = counter;
        }
        E[counter] = to;
        counter++;
    }

    for (int i = 0; i < 5; i++) {
        printf("%d, %d\n", V[i], E[i]);
    }


    return 0;

}
