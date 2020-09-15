
NVCC        = nvcc
NVCC_FLAGS  = -O3

SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=.o)
EXE = ktruss

default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

