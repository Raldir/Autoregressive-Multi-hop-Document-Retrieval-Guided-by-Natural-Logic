python3 -m src.admiral -c configs/bm25/$1.json+configs/dataset/$2.json+configs/genre/$3.json+configs/reranker/$4.json+configs/sufficiency/$5.json -k exp_name=$2/$1_$3_$4_$5_STAMMBACH_DEBUG/$6 seed=$6 continue_from_iteration=1 is_debug=True $7