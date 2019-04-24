# CSCI 563 Graduate Assignment
## Author: Clayton Kramp
### Description:
This project is a parallel implementation of BFS (Breadth First Search) algorithm.

### Usage:
To compile, type `make`.  This will build all the necessary files.

To run, type `./main [inputFile]`

If you want to generate synthetic data, type `./graphMaker` which will auto generate a graph for you called `graph.txt`.  Otherwise, import your own file.

### Input file:
The input file must look like the following:

NUM THREADS

NUM VERTICES NUM EDGES

Vertex\_To Vertex\_From Weight

### Files:
`main.cu`: the main file that runs a sequential version, a parallel version, and compares the results

`loadBalance.cu:` deploys less threads and gets at least as good results as `main`

`pure*:` pure files that do not run sequential / parallel

`graphMaker.cpp:` creates a synthetic graph

