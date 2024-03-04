futhark:
	mkdir -p lib
	futhark opencl --library -o lib/nn Main.fut
	gcc lib/nn.c -o lib/nn.so -fPIC -shared
