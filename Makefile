LOG_RD_INPUT_FILE_NAME=conv_3x3_trace.out
LOG_RD_OUTPUT_FILE_NAME=new_conv_3x3_trace.out

splay_tree.o:splay_tree.h
	g++ -g -c splay_tree.cpp

compile_dataflow:splay_tree.o
	g++ -O0 -g dataflow.cpp splay_tree.o -o dataflow

run_dataflow:
	./dataflow

compile_dataflow_test:splay_tree.o
	g++ -g -c dataflow.cpp
	g++ -O0 -g test.cpp dataflow.o splay_tree.o -o test

compile_log_rd:
	g++ -g log_rd.cpp -o log_rd

run_log_rd:
	./log_rd ${LOG_RD_INPUT_FILE_NAME} ${LOG_RD_OUTPUT_FILE_NAME}

compile_verify:splay_tree.o
	g++ -g -c dataflow.cpp
	g++ -O0 -g verify.cpp dataflow.o splay_tree.o -o verify -fopenmp

clean:
	rm -f *.o log_rd dataflow test verify
