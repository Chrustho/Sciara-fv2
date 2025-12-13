#!/bin/bash
nome="cfamo_32x8"

echo "Faccio la profilazione per OP: $nome"
# OP
nvprof --log-file "${nome}_OP.csv" --csv \
	--metrics flop_count_dp --metrics flop_count_sp --metrics flop_count_hp \
	--metrics gld_transactions --metrics gst_transactions \
	--metrics atomic_transactions \
	--metrics local_load_transactions --metrics local_store_transactions \
	--metrics shared_load_transactions --metrics shared_store_transactions \
	--metrics l2_read_transactions --metrics l2_write_transactions \
	--metrics dram_read_transactions --metrics dram_write_transactions \
	./sciara_cuda ./data/2006/2006_000000000000.cfg ./data/2006/output_2006  100 100 1.0

# rimuovo le informazioni di log di nvprof in modo da avere csv puliti 
sed -i '/^==/d' "${nome}_OP.csv"
# elapsed time 

echo "Faccio la profilazione per durata: $nome"
nvprof --log-file "${nome}_time.csv" --csv --print-gpu-summary ./sciara_cuda ./data/2006/2006_000000000000.cfg ./data/2006/output_2006  100 100 1.0

sed -i '/^==/d' "${nome}_time.csv"

echo "Finito. File generati:"
ls -lh ${nome}_*.csv
