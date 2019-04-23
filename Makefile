CXX = g++
all: sequential graphMaker main
sequential: sequential.cpp
	g++ -std=c++11 sequential.cpp -o sequential

graphMaker: graphMaker.cpp
	g++ -std=c++11 graphMaker.cpp -o graphMaker

main: main.cu
	nvcc main.cu –o main –arch=sm_35 -D_FORCE_INLINES
clean:
	rm *.o
