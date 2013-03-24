# Run from binaries directory

CXX = c++
CXXFLAGS = -std=c++11 -stdlib=libc++ -Wall -g
LDFLAGS = -I include -I include/test
OPENCV_LIBS = -L /usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc

RM = rm -f
MV = mv a.out

main_entry_point = main.cpp
main_assemblies = LogisticRegression.o
main_executable = main.lr

vpath %.hpp include include/test
vpath %.cpp src src/test

%.o: %.cpp %.hpp
	$(CXX) -c $< $(CXXFLAGS) $(LDFLAGS) -o $@

all: $(main_entry_point) $(main_assemblies)
	ctags -R .
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OPENCV_LIBS) $^
	$(MV) $(main_executable)

run: all
	./$(main_executable)

clean:
	$(RM) $(main_executable)
	$(RM) $(main_assemblies)
	$(RM) $(test_executable)
	$(RM) $(test_assemblies)
	$(RM) $(training_assemblies)
	$(RM) $(training_executable)
	$(RM) a.out
