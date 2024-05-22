lib/graia.py: graia.fut
	@echo "Graia tests running…"
	@futhark test --interpreted --no-terminal graia.fut
	@rm -f *.expected *.actual
	@echo "Graia compiling…"
	@mkdir -p lib
	@touch lib/__init__.py
	@futhark opencl --library -w -o lib/graia graia.fut
	@build_futhark_ffi lib/graia
	@mkdir -p colab_build
	@cp lib/_graia.cpython-311-x86_64-linux-gnu.so colab_build/_graia
	@echo "OK"
