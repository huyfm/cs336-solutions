%:
	@nvcc -arch=sm_86 -o % %.cu

clean:
	@find . -type f -iname "*sqlite" -delete
	@find . -type f -iname "*nsys-rep" -delete
	@find . -type f -executable -delete

.PHONY:
