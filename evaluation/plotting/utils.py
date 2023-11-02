####################################################
# This script defines some utils to analyze        #
# the output of perf_analyzer                      #
# Author: Haoran Zhao                              #
# Date: October 2023                               #
####################################################

from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import re

from typing import Union

def check_inputpath(input_path: Union[str, Path]) -> Path:
    if not isinstance(input_path, Path):
        input_path = Path(input_path)
    if not input_path.exists():
        raise Exception(f"File {input_path} not found. ")
    return input_path


def check_outputpath(output_path: Union[str, Path]) -> Path:
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def extract_numbers_GPU(row: str):
    numbers = re.findall(r"(?<=:)\d+\.\d+|(?<=:)\d+", row)
    if len(numbers) > 0:
        numbers = np.array(numbers, dtype=float)
        return numbers
    else: 
        return None

def read_perf_analyzer_output(csv_file: Union[str, Path]) -> pd.DataFrame:
    csv_file = check_inputpath(csv_file)
    df_csv = pd.read_csv(csv_file)
    df_csv.sort_values(by=['Concurrency'], inplace=True)
    df_csv.index = df_csv['Concurrency']

    # change the ave GPU utilization to float 
    for column in df_csv.columns:
        if not "GPU" in column:
            continue

        df_csv[column] = df_csv[column].apply(extract_numbers_GPU)

    return df_csv 

def plot_backend(backend_type: str, backend_results_path: Union[str, Path], 
                 var_name: str = "Inferences/Second", 
                 output_path: Union[str, Path]=None) -> (plt.Figure, plt.Axes):

    backend_results_path = check_inputpath(backend_results_path)
    if output_path is None:
        output_path = backend_results_path

    different_ins_results_path = sorted([item for item in backend_results_path.iterdir() if item.is_dir()])
    # different_ins_results_path = 
    sync_modes = ["async", "sync"]
    n_gpus_label = different_ins_results_path[0].stem.split("_")[1]
    for sync_mode in sync_modes:
        y_min = 40
        y_max = 0
        # breakpoint()
        fig, ax = plt.subplots()
        for instance_result_path in different_ins_results_path:
            if instance_result_path.stem.startswith("bad"):
                continue
            instance_label = instance_result_path.stem
            instance_label = "_".join(instance_label.split('_')[-2:])
            csv_file_pattern = f"*{instance_label}_{sync_mode}.csv"
            csv_file = sorted(instance_result_path.glob(csv_file_pattern))
            # print(instance_label)
            # print(instance_result_path)
            # print(csv_file)
            if len(csv_file) == 0:
                continue 

            df_csv = read_perf_analyzer_output(csv_file[0])
            
            # Plot
            ax.plot(df_csv['Concurrency'], df_csv[var_name], marker='o', label=instance_label)

            # Update y_min and y_max
            y_min = min(y_min, df_csv[var_name].min())
            y_max = max(y_max, df_csv[var_name].max())

        ax.set_ylim(0.9*y_min, y_max*1.3)

        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput [events/sec]")
        ax.set_title(f"GPU custom backend, {sync_mode} mode")

        ax.legend()

        fig.savefig(output_path / f"perf_vs_concurrency_{backend_type}_{n_gpus_label}_{sync_mode}.png", dpi=300)

    return fig, ax 

def plot_backend_compare(custom_backend_results: Union[str, Path], ensemble_backend_results: Union[str, Path],
                         var_name: str = "Inferences/Second",
                         output_path: Union[str, Path]=None) -> None:

    custom_backend_results = check_inputpath(custom_backend_results)
    ensemble_backend_results = check_inputpath(ensemble_backend_results)

    if output_path is None:
        output_path = check_outputpath("./compare")
    else:
        output_path = check_outputpath(output_path)
    
    different_ins_results_path = sorted([item for item in custom_backend_results.iterdir() if item.is_dir()])

    sync_modes = ["async", "sync"]
    gpus_label = different_ins_results_path[0].stem.split("_")[1]
    output_path = check_outputpath(output_path / gpus_label)

    for sync_mode in sync_modes:
        fig_all, ax_all = plt.subplots()
        for instance_result_path in different_ins_results_path:
            if instance_result_path.stem.startswith("bad"):
                continue
            custom_instance_result_path = instance_result_path 
            ensemble_instance_result_path = ensemble_backend_results / f"ensemble_{instance_result_path.stem}"
            ensemble_instance_result_path = check_inputpath(ensemble_instance_result_path)

            instance_label = instance_result_path.stem
            instance_label = "_".join(instance_label.split('_')[-2:])
            csv_file_pattern = f"*{instance_label}_{sync_mode}.csv"

            custom_csv_file = sorted(custom_instance_result_path.glob(csv_file_pattern))
            ensemble_csv_file = sorted(ensemble_instance_result_path.glob(csv_file_pattern))

            custom_df_csv = read_perf_analyzer_output(custom_csv_file[0])
            ensemble_df_csv = read_perf_analyzer_output(ensemble_csv_file[0])

            # Plot
            ax_all.plot(custom_df_csv['Concurrency'], custom_df_csv[var_name], marker='o', label=f"{instance_label} custom backend")
            ax_all.plot(ensemble_df_csv['Concurrency'], ensemble_df_csv[var_name], marker='^', label=f"{instance_label} ensemble scheduler")

            fig, ax = plt.subplots()
            ax.plot(custom_df_csv['Concurrency'], custom_df_csv[var_name], marker = "o", label = "custom backend")
            ax.plot(ensemble_df_csv['Concurrency'], ensemble_df_csv[var_name], marker = "^", label="ensemble scheduler")

            ax.set_xlabel("Concurrency")
            ax.set_ylabel(var_name)
            ax.set_title(f"{var_name} vs Concurrency, {instance_label}, {sync_mode} mode")

            ax.legend()
            fig.savefig(output_path / f"perf_vs_concurrency_{instance_label}_{sync_mode}.png", dpi=300)
            
        ax_all.set_xlabel("Concurrency")
        ax_all.set_ylabel(var_name)
        ax_all.set_title(f"{var_name} vs Concurrency, {sync_mode} mode")
        ax_all.legend()
        fig_all.savefig(output_path / f"perf_vs_concurrency_all_{sync_mode}.png", dpi=300)

        
def exatract_throughput_vs_instances(backend_results_path, pattern = r'(\d+)insts', sync_mode = "sync", var_name = "Inferences/Second", n_instance_threshold = 10):
    backend_results = {
        "n_instances": [],
        "Throughput_mean": [],
        "Throughput_std": [], 
        "GPU Utilization": [],
    }

    different_ins_results_path = sorted([item for item in backend_results_path.iterdir() if item.is_dir()])
    for instance_result_path in different_ins_results_path:
        
        match = re.search(pattern, instance_result_path.stem)
        if match:
            n_instance = int(match.group(1)) 
        else:
            raise ValueError("No instance number found in the path.") 

        backend_results["n_instances"].append(n_instance)
        csv_file_pattern = f"*_{sync_mode}.csv"
        csv_file = sorted(instance_result_path.glob(csv_file_pattern))[0]

        pd_csv = read_perf_analyzer_output(csv_file)

        pd_saturated = pd_csv.query(f"Concurrency >= {n_instance_threshold}")
        backend_results["Throughput_mean"].append(pd_saturated[var_name].mean())
        backend_results["Throughput_std"].append(pd_saturated[var_name].std())
        
        gpu_util = pd_saturated['Avg GPU Utilization'].apply(extract_numbers_GPU).mean()
        backend_results["GPU Utilization"].append(gpu_util)

    return backend_results


timing_items = ['Client Send', 'Network+Server Send/Recv', 'Server Queue', 'Server Compute Input', 'Server Compute Infer', 'Server Compute Output', 'Client Recv']

def plot_timing_breakout(df: pd.DataFrame, timing_items: list = timing_items, fig = None, ax = None):
    df = df.copy()
    df['total'] = df[timing_items].sum(axis=1)

    # draw a stacked bar plot of the total time
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    bottom = np.zeros(len(df))
    for timing_item in timing_items:
        ax.bar(df['Concurrency'], df[timing_item], bottom=bottom, label=timing_item)
        bottom += df[timing_item]
    ax.scatter(df.index, df['total'], marker='x', color='black', label='Total')    

    ax.set_ylabel('Time (usec)')
    ax.legend()

    return fig, ax