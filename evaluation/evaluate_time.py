import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
import numpy as np 
sns.set_theme()

import logging
logging.basicConfig(level=logging.INFO)

def run_time_command(script_name, num_runs=5):
    timing_results = []
    if type(script_name) == str:
        script_name = [script_name]

    for _ in range(num_runs):
        cmd = ['time', '-p', 'bash'] + script_name
        # Run the command and capture stderr (where time outputs its results)
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

        # Check for errors, return error if there is one
        if result.returncode != 0:
            raise Exception(result.stderr)
        
        # Extract the real, user, and sys time from the output
        for line in result.stderr.split('\n'):
            if line.startswith('real'):
                real_time = float(line.split()[1])
            elif line.startswith('user'):
                user_time = float(line.split()[1])
            elif line.startswith('sys'):
                sys_time = float(line.split()[1])

        timing_results.append({
            'real': real_time,
            'user': user_time,
            'sys': sys_time
        })
    
    return timing_results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-runs', help='Number of times to run the script', type=int, default=10)
    parser.add_argument('--use-more', help='Test on more events(5000)', action='store_true')

    args = parser.parse_args()
    scripts = ['inference_gpu_direct.sh', 'inference_gpu_triton.sh']

    num_runs = args.num_runs 

    comparison_results = dict.fromkeys(scripts)
    for script in scripts:
        if script == 'inference_gpu_triton.sh':
            input('Please start the triton server and press enter to continue')

            # test if the triton server is running by using ps aux | grep [t]ritonserver
            # if not, raise an error
            # TODO: Not sure how to do this. 
            # two containers are running, one is the tritonserver, the other is the client
            # but the client is not running the tritonserver, so the ps aux | grep [t]ritonserver 
            # will not return anything
            # cmd = ['pgrep', '-f', 'tritonserver']
            # result = subprocess.run(cmd)
            # if result.returncode != 0:
            #     raise Exception('Triton server is not running')
            
        comparison_results[script] = dict.fromkeys(['mean', 'std'])

        logging.info(f"Running \n    {script}    \n    for {num_runs} times")

        if args.use_more:
            logging.info(f"Running on more events")
            script_more = [script, '/workspace/exatrkx_pipeline/datanmodels/lrt/more/']
            results = run_time_command(script_more, num_runs)
        else:
            results = run_time_command(script, num_runs)

        # convert results to pandas dataframe 
        df = pd.DataFrame(results)

        # add a row for mean and standard deviation
        df.loc['mean'] = df.mean()
        df.loc['std'] = df.std()

        comparison_results[script]['mean'] = df['real']['mean']
        comparison_results[script]['std'] = df['real']['std']

        # make a scatter plot of the results
        # x-axis is the run number, plus mean and std as yerr 
        # y-axis is the real time
        fig, ax = plt.subplots()
        ax.scatter(np.arange(num_runs+1), df['real'][:-1], label='real')
        ax.errorbar(x=num_runs, y=df['real']['mean'], yerr=df['real']['std'], fmt='o', label='mean')
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, num_runs+2)))
        ax.set_xticklabels(list(map(str, np.arange(num_runs+1))) + ['mean'])

        ax.set_xlabel('Run Number')
        ax.set_ylabel('Real Time (s)')
        ax.set_title(f'Real Time for {num_runs} Runs of {script}')
        fig.savefig(f"results_{script.split(sep='.')[0]}.png")
    
    
    # dump comparison_results 
    # import pickle
    # with open('gpu_comparison.pkl', 'wb') as f:
    #     pickle.dump(comparison_results, f)

    # compare two scripts 
    fig, ax = plt.subplots()
    colors = ['red', 'blue']
    for i, script in enumerate(comparison_results.keys()):
        ax.errorbar(x=i, 
                    y=comparison_results[script]['mean'], 
                    yerr=comparison_results[script]['std'], 
                    fmt='o', color=colors[i], capsize=10, label=script)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0, len(scripts))))
    ax.set_xticklabels(list(comparison_results.keys()))
    ax.legend()
    ax.set_ylabel('Real Time (s)')
    ax.set_title('GPU Direct vs Triton Average Inference Time')
    fig.savefig('gpu_comparison.png')


if __name__ == '__main__':
    main()