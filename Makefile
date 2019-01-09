SRC = ./src
OUTPUT = ./output
CLFLAGS = -O3 -std=c++17
CVLIBS = `pkg-config opencv`
CVFLAGS = `pkg-config --libs --cflags opencv`

all: out

out: $(SRC)/main.cpp  $(SRC)/*
	g++ $(CLFLAGS) $(SRC)/*.cpp -o $(OUTPUT)/out $(CVFLAGS)

clean:
	rm $(OUTPUT)/*
