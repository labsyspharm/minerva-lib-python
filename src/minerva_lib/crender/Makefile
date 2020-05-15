CC=gcc
CFLAGS=-fPIC -O3 -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -mavx -ffast-math -funsafe-math-optimizations -fno-math-errno

all: render crender.so test_render

crender.so : render.o
	$(CC) $(CFLAGS) -shared -Wl,-soname,crender.so -o crender.so render.o

render:
	$(CC) $(CFLAGS) render.c -c

test_render:
	$(CC) $(CFLAGS) -o test_render render.c test_render.c

clean:
	rm -f *.o *.so

