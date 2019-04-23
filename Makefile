CXX = g++
all: sequential graphMaker
sequential: sequential.cpp
	g++ -std=c++11 sequential.cpp -o sequential

graphMaker: graphMaker.cpp
	g++ -std=c++11 graphMaker.cpp -o graphMaker

clean:
	rm *.o
