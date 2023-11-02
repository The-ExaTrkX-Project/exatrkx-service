#!/usr/bin/env python

#=========================================
#
# Title: triton_metrics.py
# Author: Andrew Naylor
# Date: Jun 23
# Brief: Print out metrics to terminal
#
# Usage: python triton_metrics.py server_address
#
#=========================================

import sys
import requests
from prometheus_client.parser import text_string_to_metric_families  # pip install prometheus_client
from time import sleep
from copy import deepcopy
import numpy as np
import pandas as pd
import time


models = ['embed', 'frnn', 'filter', 'applyfilter', 'gnn', 'wcc', 'exatrkx']

delta_func = lambda x: x[-1] - x[0]
diff_func = lambda x: f'{delta_func(x):.0f}'
avg_func = lambda x, y: f'{(delta_func(x)/(1000)/delta_func(y)):.4f} ms per event with {delta_func(y)} events.'

interested_metrics = {
    'nv_inference_request_success': {'name': 'Success Count', 'func': diff_func},
    'nv_inference_request_duration_us': {'name': 'Request Time', 'func': avg_func},
    'nv_inference_queue_duration_us': {'name': 'Queue Time', 'func': avg_func},
    'nv_inference_compute_input_duration_us': {'name': 'Compute Input Time', 'func': avg_func},
    'nv_inference_compute_infer_duration_us': {'name': 'Compute Time', 'func': avg_func},
    'nv_inference_compute_output_duration_us': {'name': 'Compute Output Time', 'func': avg_func}
}

metrics_labels = interested_metrics.keys()
models_empty_dict = {k: [] for k in models}
metrics_data = {k: deepcopy(models_empty_dict) for k in interested_metrics}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Triton Metrics')
    parser.add_argument('--ip', type=str, default='localhost:8002', help='Triton Metrics IP')
    parser.add_argument('--timeout', type=int, default=20, help='Timeout in seconds')
    parser.add_argument('--freq', type=float, default=0.000001, help='Polling frequency in seconds')
    parser.add_argument('--outname', type=str, default='metrics', help='Output file name')
    args = parser.parse_args()

    timeout = args.timeout
    triton_metrics_ip = args.ip
    poll_freq = args.freq
    outname = args.outname

    request_count_name = 'nv_inference_request_success'
    main_model = 'exatrkx'
    request_time_name = 'nv_inference_request_duration_us'
    execution_time_name = 'nv_inference_compute_infer_duration_us'

    triton_metrics_path = f'http://{triton_metrics_ip}/metrics'

    # Loop until timeout
    print("<pulling data>")
    start_time = time.time()
    elapsed_time = 0
    while elapsed_time < timeout:
        res = requests.get(triton_metrics_path)
        res_data = res.content.decode('utf-8')
        _interested_metrics = interested_metrics.copy()

        # check if the request count is increased
        save_data = True
        if len(metrics_data[request_count_name][main_model]) > 0:
            for i in text_string_to_metric_families(res_data):
                if i.name == 'nv_inference_request_success':
                    for j in i.samples:
                        if j.labels['model'] == main_model:
                            if metrics_data[i.name][j.labels['model']][-1] == j.value:
                                save_data = False
                                break
                    break

        if save_data:
            for i in text_string_to_metric_families(res_data):
                # only save the metrics when the request count is increased.

                if i.name in metrics_labels:
                    # print(f'<> Processing metric {i.name}')
                    for j in i.samples:
                        # print(f"{j.labels['model']} - {j.value}")
                        val = int(j.value) if i.name == request_count_name else j.value
                        metrics_data[i.name][j.labels['model']].append(val)

                    _interested_metrics.pop(i.name)

                if len(_interested_metrics) == 0:
                    break
        sleep(poll_freq)
        elapsed_time = time.time() - start_time

    # Print out main metrics
    for k, v in metrics_data.items():
        print(f'<> {interested_metrics[k]["name"]}')
        for m in models:
            if k == request_count_name:
                # only print out the number of requests
                print(f'    - {m} - {interested_metrics[k]["func"](v[m])} - {len(v[m])}', v[m])
            else:
                # print out the average time per event
                print(f'    - {m} - {interested_metrics[k]["func"](v[m], metrics_data[request_count_name][m])}')

    # save these metrics into two csv files: request time and execution time
    # columns are the models and rows are the events
    # we need to ensure that the events are the same for all models
    # and entries are the time in seconds
    # filtering and applyfilter should be combined into one

    # save the request time
    num_entries = len(metrics_data[request_count_name][main_model])
    request_time_dict = {k: {} for k in models}
    execution_time_dict = {k: {} for k in models}
    for model in models:
        request_time_dict[model] = dict(zip(metrics_data[request_count_name][model], metrics_data[request_time_name][model]))
        execution_time_dict[model] = dict(zip(metrics_data[request_count_name][model], metrics_data[execution_time_name][model]))

    request_time = []
    execution_time = []

    pre_evtid = metrics_data[request_count_name][main_model][0]

    for idx in range(num_entries):
        evtid = metrics_data[request_count_name][main_model][idx]
        if evtid == pre_evtid:
            continue
        num_evts = evtid - pre_evtid
        # make sure this event in all models
        try:
            request_time.append([
                (request_time_dict[model][evtid] - request_time_dict[model][pre_evtid]) / 1000_000 / num_evts
                for model in models
                ] + [num_evts])
            execution_time.append([
                (execution_time_dict[model][evtid] - execution_time_dict[model][pre_evtid]) / 1000_000 / num_evts
                for model in models
                ] + [num_evts])
        except KeyError:
            print(f'event {evtid} is not in all models')
        pre_evtid = evtid

    if len(request_time) == 0:
        print('no event is found in common among all models. Use a smaller frequency.')
        sys.exit(1)

    def save_timing_info(timing_info, outname):
        timing_info = np.array(timing_info)
        df = pd.DataFrame(timing_info, columns=models + ['num_evts'])
        df['filter'] = df['filter'] + df['applyfilter']
        df = df.drop(columns=['applyfilter'])
        df = df.astype({'num_evts': 'int32'})
        outname = f'{outname}.csv'
        df.to_csv(outname, index=False,
                  header=True, sep=',', float_format='%.4f')

    save_timing_info(request_time, f'{outname}_request_time')
    save_timing_info(execution_time, f'{outname}_execution_time')
