TRACE_DIN=../MAC_data/Conv_dataflow/AlexNet_13_13_8_4_trace.din
CACHE_SIM_RESULT=AlexNet_cache_sim/256KB_4B/Weight_Reuse/CONV5/conv_13_13_8_4_cache_sim.out

./dineroIV -l1-isize 8k -l1-iassoc 1 -l1-ibsize 16 -l1-irepl f -l1-dsize 256k -l1-dassoc 65536 -l1-dbsize 4 -l1-drepl l -l1-dwback a -l1-dccc -l1-iccc -informat d <${TRACE_DIN}> ${CACHE_SIM_RESULT}
