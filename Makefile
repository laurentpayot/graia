futhark:
	mkdir -p lib
	futhark opencl --library -o lib/nn Main.fut
