CC=gcc
CFLAGS=-fPIC -O3 -march=haswell -ffast-math -funsafe-math-optimizations -fno-math-errno

all: render crender.so test_render

crender.so : render.o
	$(CC) $(CFLAGS) -shared -Wl,-soname,crender.so -o crender.so render.o

render:
	$(CC) $(CFLAGS) render.c -c

test_render: test_render.c
	$(CC) $(CFLAGS) -o test_render render.c test_render.c

clean:
	rm -f *.o *.so test_render

test: test_render
	./test_render

