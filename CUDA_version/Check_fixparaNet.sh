#!/bin/bash

sudo /usr/local/cuda-12.2/nsight-compute-2023.2.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct --launch-count 120 --launch-skip 3000 --export profile_aligen_scale70 ./EI_net_cuda_fixpara_scale40