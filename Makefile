SRC = ./src
OUTPUT = ./output
CLFLAGS = -O3
OPENCV = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_objdetect -lopencv_imgproc -lopencv_ml


all: out

out: $(SRC)/main.cpp  $(SRC)/hog_visualization.cpp $(SRC)/hog_visualization.h
	g++ $(CLFLAGS) $(OPENCV) $(SRC)/main.cpp $(SRC)/hog_visualization.cpp -o $(OUTPUT)/out

clean:
	rm $(OUTPUT)/*
