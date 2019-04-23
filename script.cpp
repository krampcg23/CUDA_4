#include <iostream>
#include <sstream>
#include <fstream>
#include <set>

using namespace std;

int main(int argc, char* argv[]) {
    string file = argv[1];
    ifstream f(file);
    int counter = 0;
    set<int> s;
    while(!f.eof()) {
        string line;
        getline(f, line);
        counter++;
        stringstream ss(line);
        int r, c, v;
        ss >> r >> c >> v;
        s.insert(r);
        s.insert(c);
    }
    cout << "Edges: " << counter << endl;
    cout << "Vertices: " << s.size() << endl;
    return 0;
}
