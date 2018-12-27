SRC = ./src
OUTPUT = ./output
CLFLAGS = -O3
CVLIBS = `pkg-config opencv`
CVFLAGS = `pkg-config --libs --cflags opencv`

all: out

out: $(SRC)/main.cpp  $(SRC)/*
	g++ $(CLFLAGS) $(CVFLAGS) $(SRC)/main.cpp $(SRC)/hog_visualization.cpp -o $(OUTPUT)/out

clean:
	rm $(OUTPUT)/*
