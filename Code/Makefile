PKG_NAME := $(shell pkg-config --exists Magick++ && echo "Magick++" || echo "ImageMagick++")

CXX      := mpicxx
CXXFLAGS := -DOMPI_SKIP_MPICXX -std=c++23 -O3 -Wall -Wextra -Werror $(shell pkg-config --cflags $(PKG_NAME))
LDFLAGS  := $(shell pkg-config --libs $(PKG_NAME)) -lpthread

TARGET   := main
SRCS     := main.cc
#HDRS     := ThreadSafeQueue.hpp
OBJS     := $(SRCS:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cc $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean