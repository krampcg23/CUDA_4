#include <iostream>
#include <fstream>

using namespace std;

int main() {

    int vertices = 10;
    int edges = 30;

    srand(time(NULL));

    ofstream file("graph.txt");
    file << vertices << " " << edges << endl;

    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < edges / vertices; j++) {
            file << rand() % 10 + 1 << " " << i+1 << " " << 1 << endl;
        }
    }

    return 0;
}


