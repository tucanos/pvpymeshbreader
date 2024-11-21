CC=gcc
RM=rm -f
CFLAGS=-fpic -Wall -g -O0 -std=c99 -Isrc/
LDFLAGS=-shared
SRCS=src/libmeshb7.c src/meshb_io.c

OBJS=$(subst .c,.o,$(SRCS))

TARGET= _meshb_io.so

all: $(TARGET)

run: $(TARGET)
	./test

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)
    
clean:
	$(RM) $(OBJS) $(TARGET)