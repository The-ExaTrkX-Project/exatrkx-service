#!/usr/bin/env python

#=========================================
# 
# Title: test_triton_model.py
# Author: Andrew Naylor
# Date: Jun 23
# Brief: Test triton models
#
# Usage: python test_triton_model.py [triton_model] [inputs] -t/--triton_address [ip_address]
#
#=========================================

import os
import argparse

from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient
import numpy as np

#Configs
models = {
    'exatrkx': {'inputs': ['FEATURES'], 'type': [np.single], 'outputs': ['LABELS']},
    'applyfilter': {'inputs': ['FILTER_SCORES', 'EDGE_LIST'], 'type': [np.single, np.int64], 'outputs': ['EDGE_LIST_AFTER_FILTER']},
    'embed': {'inputs': ['INPUT__0'], 'type': [np.single], 'outputs': ['OUTPUT__0']},
    'filter': {'inputs': ['INPUT__0', 'INPUT__1'], 'type': [np.single, np.int64], 'outputs': ['OUTPUT__0']},
    'frnn': {'inputs': ['INPUT0'], 'type': [np.single], 'outputs': ['OUTPUT0']},
    'gnn': {'inputs': ['INPUT__0', 'INPUT__1'], 'type': [np.single, np.int64], 'outputs': ['OUTPUT__0']},
    'wcc': {'inputs': ['INPUT0', 'INPUT1'], 'type': [np.int64, np.single], 'outputs': ['OUTPUT0']}
}

if __name__ == "__main__":
    #Args
    parser = argparse.ArgumentParser(
        description='Test triton models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('triton_models', type=str, nargs=1,
                        help='Triton models', choices=models.keys())
    parser.add_argument('inputs', type=str, nargs='+',
                        help='Input files')
    parser.add_argument('-t', '--triton_address', type=str, 
                        help='Triton server address', default='localhost:8001')

    args = parser.parse_args()
    triton_model = args.triton_models[0]

    print(f'Executing model {triton_model} on {args.triton_address} server')
    print(f'    - Inputs:')

    #Triton inference call
    with grpcclient.InferenceServerClient(args.triton_address) as client:
        #Setup
        grpc_inputs = []
        for i, f in enumerate(args.inputs):
            in_file = open(f, 'rb')
            in_data = np.loadtxt(in_file, delimiter=',').astype(models[triton_model]['type'][i])
            if 'applyfilter' == triton_model and 'FILTER_SCORES' == models[triton_model]["inputs"][i]:
                in_data = in_data.reshape(-1, 1)
            print(f'           +  {f} {in_data.shape} {in_data.dtype} {models[triton_model]["inputs"][i]}')

            grpc_inputs.append(
                grpcclient.InferInput(
                    models[triton_model]['inputs'][i], in_data.shape,
                    np_to_triton_dtype(in_data.dtype)
                )
            )
            grpc_inputs[i].set_data_from_numpy(in_data)
        grpc_outputs = [grpcclient.InferRequestedOutput(label) for label in models[triton_model]['outputs']]

        #Process response
        response = client.infer(triton_model,
                                grpc_inputs,
                                request_id=str(1),
                                outputs=grpc_outputs)

        result = response.get_response()

        #Outputs
        print(f'    - Outputs:')
        for label in models[triton_model]['outputs']:
            out_data = response.as_numpy(label)
            out_file = f'{triton_model}_output_{label}.csv'
            print(f'           +  {out_file} {out_data.shape} {out_data.dtype} {label}')
            print(out_data)

            #Save
            np.savetxt(out_file, out_data, delimiter=",")