LOG_RD_INPUT_FILE_NAME=conv_3x3_trace.out
LOG_RD_OUTPUT_FILE_NAME=new_conv_3x3_trace.out

splay_tree.o:splay_tree.h
	g++ -g -c splay_tree.cpp

compile_dataflow:splay_tree.o
	g++ -O0 -g dataflow.cpp splay_tree.o -o dataflow

compile_test:splay_tree.o
	g++ -g -c dataflow.cpp
	g++ -O0 -g test.cpp dataflow.o splay_tree.o -o test

compile_log_rd:
	g++ -g log_rd.cpp -o log_rd

run_log_rd:
	./log_rd ${LOG_RD_INPUT_FILE_NAME} ${LOG_RD_OUTPUT_FILE_NAME}

compile_estimation:
	g++ -g estimation.cpp -o estimation

compile_conv_mali:splay_tree.o
	g++ -g -c dataflow.cpp
	g++ -g -c conv_mali.cpp
	g++ -g conv_mali_test.cpp conv_mali.o dataflow.o splay_tree.o -o conv_gpu -lOpenCL

compile_conv_mali_on_MAC:splay_tree.o
	g++ -g -c dataflow.cpp
	g++ -g -c conv_mali.cpp
	g++ -g conv_mali_test.cpp conv_mali.o dataflow.o splay_tree.o -o conv_gpu -framework opencl

clean:
	rm -f *.o log_rd dataflow test verify conv_gpu
