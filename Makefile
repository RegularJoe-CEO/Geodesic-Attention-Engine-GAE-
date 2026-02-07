NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90 --use_fast_math

TARGET = waller_benchmark

all: $(TARGET)

$(TARGET): src/waller_operator.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

