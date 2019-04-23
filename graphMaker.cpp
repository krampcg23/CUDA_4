#include <iostream>
#include <fstream>

using namespace std;

int main() {

    int vertices = 30;
    int edges = 60;

    srand(time(NULL));

    ofstream file("graph.txt");
    file << vertices << " " << edges << endl;

    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < edges / vertices; j++) {
            file << rand() % vertices + 1 << " " << i+1 << " " << 1 << endl;
        }
    }

    return 0;
}


