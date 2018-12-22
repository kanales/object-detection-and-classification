SRC = ./src
OUTPUT = ./output
CLFLAGS = -O3
CVLIBS = `pkg-config --libs opencv`
CVFLAGS = `pkg-config --cflags opencv`

all: out

out: $(SRC)/main.cpp  $(SRC)/*
	g++ $(CLFLAGS) $(CVFLAGS) $(CVLIBS) $(SRC)/main.cpp $(SRC)/hog_visualization.cpp -o $(OUTPUT)/out

clean:
	rm $(OUTPUT)/*
