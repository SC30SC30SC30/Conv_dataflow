TRACE_DIN=../MAC_data/Conv_dataflow/Paper_Result/AlexNet/conv_1_1_4_256_mem_trace.din
CACHE_SIM_RESULT=AlexNet_cache_sim/CONV5/conv_1_1_4_256_cache_sim.out

./dineroIV -l1-isize 8k -l1-iassoc 1 -l1-ibsize 16 -l1-irepl f -l1-dsize 256k -l1-dassoc 4096 -l1-dbsize 64 -l1-drepl l -l1-dwback a -l1-dccc -l1-iccc -informat d <${TRACE_DIN}> ${CACHE_SIM_RESULT}
