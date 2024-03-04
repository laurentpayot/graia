futhark:
	mkdir -p lib
	futhark opencl --library -o lib/nn Main.fut
	gcc -shared -o lib/nn.so -fPIC lib/nn.c
