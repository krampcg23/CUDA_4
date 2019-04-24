#include <iostream>
#include <fstream>

using namespace std;

int main() {

    int vertices = 10000000;
    int edges = 50000000;

    srand(time(NULL));

    ofstream file("DataSet/graph.txt");
    file << 1024 << endl;
    file << vertices << " " << edges << endl;

    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < edges / vertices; j++) {
            file << rand() % vertices + 1 << " " << i+1 << " " << 1 << endl;
        }
    }

    return 0;
}


