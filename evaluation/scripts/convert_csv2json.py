####################################################
# This script converts the CSV file to a JSON file #
# for the input of perf_analyzer                   #
# Author: Haoran Zhao                              #
# Date: October 2023                               #
# Example:                                         #
# python convert_csv2json.py --input in_e1000.csv  #
####################################################

from pathlib import Path 
import pandas as pd
import numpy as np 
import json 

import argparse
from typing import Union
from tqdm import tqdm

def read_csv(input_file: Union[str, Path]) -> pd.DataFrame:
    """Read the CSV file into a pandas DataFrame.

    Args:
        input_file (Union[str, Path]): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    input_file_path = Path(input_file) if isinstance(input_file, str) else input_file

    if not input_file_path.exists():
        raise FileNotFoundError(f"{input_file_path} does not exist.")

    df = pd.read_csv(input_file_path, names=['x', 'y', 'z'])

    return df

def csv2json(user_input: Union[str, Path]) -> dict:
    """Convert the CSV file/files to a JSON file.

    Args:
        user_input (Union[str, Path]): user input, can be a csv file or a folder containing CSV files.

    Raises:
        FileNotFoundError: if the input file does not exist, raise FileNotFoundError.
        ValueError: if the input is not a file or folder, raise ValueError.

    Returns:
        dict: a dictionary containing the data.
    """
    user_input_path = Path(user_input) if isinstance(user_input, str) else user_input 

    if not user_input_path.exists():
        raise FileNotFoundError(f"{user_input_path} does not exist.")
    
    output_data_structure = {"data" : []}
    # check if the input is a file or folder 
    if user_input_path.is_file():
        df = read_csv(user_input_path)
        flattened_list = df.values.flatten().tolist()
        output_data_structure["data"].append({"FEATURES": {"content": flattened_list, "shape": list(df.shape)}})

    elif user_input_path.is_dir():
        for csv_file in tqdm(user_input_path.glob("*.csv")):
            df = read_csv(csv_file)
            flattened_list = df.values.flatten().tolist()
            output_data_structure["data"].append({"FEATURES": {"content": flattened_list, "shape": list(df.shape)}})

    else:
        raise ValueError(f"{user_input_path} is not a file or folder.")

    output_json = user_input_path.parent / f'{user_input_path.stem}.json'
    with open(output_json, 'w') as json_file:
        json.dump(output_data_structure, json_file, indent=4)
    
    return output_data_structure

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert CSV to JSON')
    parser.add_argument('--input', type=str, help='Input CSV file, defalut is in_e1000.csv', default="/workspace/exatrkx_pipeline/datanmodels/in_e1000.csv")
    args = parser.parse_args()
    user_input = args.input

    csv2json(user_input)