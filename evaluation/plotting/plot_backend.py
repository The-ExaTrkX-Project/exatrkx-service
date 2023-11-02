####################################################
# This script analyze the perf_analyzer            #
# results and visualize them                       #
# Author: Haoran Zhao                              #
# Date: October 2023                               #
####################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path

import mplhep as hep
hep.style.use(hep.style.ATLAS)

from utils import plot_backend, plot_backend_compare

custom_1gpus = "/pscratch/sd/h/hrzhao/Projects/exatrkx-service/evaluate/slurm/good-1gpu-slurm-16730083" 
custom_2gpus = "/pscratch/sd/h/hrzhao/Projects/exatrkx-service/evaluate/slurm/good-2gpu-slurm-16733713"
ensemble_1gpus = "/pscratch/sd/h/hrzhao/Projects/exatrkx-service/evaluate_ensemble/slurm/good-1gpu-slurm-16748159"
ensemble_2gpus = "/pscratch/sd/h/hrzhao/Projects/exatrkx-service/evaluate_ensemble/slurm/slurm-16748337"
# plot_backend(backend_type="custom", 
#              backend_results_path=custom_1gpus)

# plot_backend(backend_type="custom", 
#              backend_results_path=custom_2gpus)

# plot_backend(backend_type="ensemble", 
#              backend_results_path=ensemble_1gpus)

# plot_backend(backend_type="ensemble", 
#              backend_results_path=ensemble_2gpus)

plot_backend_compare(custom_backend_results = custom_1gpus,
                     ensemble_backend_results= ensemble_1gpus)

plot_backend_compare(custom_backend_results = custom_2gpus,
                    ensemble_backend_results= ensemble_2gpus)