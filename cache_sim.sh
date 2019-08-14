TRACE_DIN=trace.din

./dineroIV -l1-isize 8k -l1-iassoc 1 -l1-ibsize 16 -l1-irepl f -l1-dsize 64k -l1-dassoc 1 -l1-dbsize 64 -l1-drepl l -l1-dwback a -l1-dccc -l1-iccc -informat d <${TRACE_DIN}> conv_cache_sim.out
