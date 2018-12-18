SRC = ./src
OUTPUT = ./output
CLFLAGS = -O3 -lopencv_core -lopencv_imgcodecs -lopencv_highgui

all: out

out: $(SRC)/main.cpp
	g++ $(CLFLAGS) $(SRC)/main.cpp -o $(OUTPUT)/out

clean:
	rm $(OUTPUT)/*
