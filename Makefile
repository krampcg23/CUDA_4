CXX = g++
all: sequential
sequential: sequential.cpp
	g++ -std=c++11 sequential.cpp -o sequential

clean:
	rm *.o
