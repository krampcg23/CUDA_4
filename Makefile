CXX = g++
all: pureParallel graphMaker main pureSerial loadBalance

loadBalance: loadBalance.cu
	nvcc -std=c++11 loadBalance.cu -o loadBalance -arch=sm_35 -D_FORCE_INLINES

pureSerial: pureSerial.cpp
	g++ -std=c++11 pureSerial.cpp -o serial

graphMaker: graphMaker.cpp
	g++ -std=c++11 graphMaker.cpp -o graphMaker

main: main.cu
	nvcc -std=c++11 main.cu -o main -arch=sm_35 -D_FORCE_INLINES

pureParallel: pureParallel.cu
	nvcc -std=c++11 pureParallel.cu -o parallel -arch=sm_35 -D_FORCE_INLINES
clean:
	rm *.o main script sequential graphMaker serial parallel loadBalance
