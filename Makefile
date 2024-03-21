lib/graia.py: graia.fut
	@echo "Graia tests running…"
	@futhark test --interpreted --no-terminal graia.fut
	@rm -f *.expected *.actual
	@echo "Graia compiling to Python OpenCL library…"
	@mkdir -p lib
	@touch lib/__init__.py
	@futhark pyopencl --library -w -o lib/graia graia.fut
	@echo "OK"
