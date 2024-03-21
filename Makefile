lib/graia.py: graia.fut
	@echo "Graia tests running…"
	@futhark test -i --pass-option=-w graia.fut
	@echo "Graia compiling to Python OpenCL library…"
	@mkdir -p lib
	@touch lib/__init__.py
	@futhark pyopencl --library -w -o lib/graia graia.fut
	@echo "OK"
