#
# Matrix Multiply example makefile
#

ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -std=c99 -ffast-math

LIBS = -lm -lOpenCL -fopenmp

COMMON_DIR = ../common

MMUL_OBJS = wtime.o
EXEC = mult


# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -lm -framework OpenCL 
endif


all: $(EXEC)

mult: $(MMUL_OBJS) matmul.c matrix_lib.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $(EXEC)

wtime.o: $(COMMON_DIR)/wtime.c
	$(CC) -c $^ $(CCFLAGS) -o $@

.c.o:
	$(CC) -c $< $(CCFLAGS) -o $@


clean:
	rm -f $(MMUL_OBJS) $(EXEC)
