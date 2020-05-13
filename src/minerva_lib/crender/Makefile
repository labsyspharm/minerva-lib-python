CC=gcc
CFLAGS=-fPIC -O3 -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -mavx -ffast-math -funsafe-math-optimizations -fno-math-errno

crender.so : render.o
	gcc -shared -Wl,-soname,crender.so -o crender.so render.o

render: render.o
	$(CC) -o render render.o
