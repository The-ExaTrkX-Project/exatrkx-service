#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <filesystem>
#include <unistd.h>
#include <limits.h>

#include "tbb/parallel_for_each.h"
#include "tbb/task_scheduler_init.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <mutex>

#include "ExaTrkXTrackFinding.hpp"

namespace fs = std::filesystem;

void processInput(std::string file_path, std::vector<float>& input_tensor_values){
    input_tensor_values.clear();

    std::ifstream f (file_path);   /* open file */
    if (!f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + file_path).c_str());
    }
    std::string line;                    /* string to hold each line */
    while (getline (f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        //array.push_back (row);          /* add row to array */
        input_tensor_values.insert (input_tensor_values.end(),row.begin(),row.end());  
    }
    f.close();
}

void dumpTrackCandidate(const std::vector<std::vector<int> >& trackCandidates) {
    int idx = 0;
    for (const auto& track_candidate : trackCandidates) {
        std::cout << "Track candidate: " << idx++ << "--> ";
        for (const auto& id : track_candidate) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
    }
}

// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    int server_type = 0;
    std::string input_file_path = "datanmodels/in_e1000.csv";
    int opt;
    bool help = false;
    bool verbose = false;
    int nthreads = 1;
    std::string model_path("datanmodels");

    while ((opt = getopt(argc, argv, "vht:d:m:")) != -1) {
        switch (opt) {
            case 'd':
                input_file_path = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 't':
                nthreads = atoi(optarg);
                break;
            case 'm':
                model_path = optarg;
                break;
            case 'h':
                help = true;
            default:
                fprintf(stderr, "Usage: %s [-hv] [-d input_file_path] [-t number_of_threads] \n", argv[0]);
                if (help) {
                    std::cerr << " -d: input data/directory" << std::endl;
                    std::cerr << " -t: number of threads" << std::endl;
                    std::cerr << " -v: verbose" << std::endl;
                    std::cerr << " -m: model path" << std::endl;
                }
            exit(EXIT_FAILURE);
        }
    }

    // start tbb scheduler
    tbb::task_scheduler_init init(nthreads);

    std::cout << "Input file: " << input_file_path << std::endl;
    

    std::unique_ptr<ExaTrkXTrackFinding> infer;
    ExaTrkXTrackFinding::Config config{model_path, verbose};
    infer = std::make_unique<ExaTrkXTrackFinding>(config);

    std::cout << "Running Inference with local GPUs" << std::endl;
    
    const fs::path filepath(input_file_path);
    std::error_code ec;

    // input containers
    std::vector<std::vector<float> > input_tensor_values;
    std::vector<std::vector<int> > spacepoint_ids;
    int64_t spacepointFeatures = 3;

    if (fs::is_directory(filepath, ec)) {
        // read all files in the directory
        // and save the spacepoints in a vector
        for (const auto& entry : fs::directory_iterator(filepath)) {
            if (entry.path().extension() == ".csv") {
                std::vector<float> input_tensor_values_temp;
                processInput(entry.path().string(), input_tensor_values_temp);
                input_tensor_values.push_back(input_tensor_values_temp);


                int numSpacepoints = input_tensor_values_temp.size()/spacepointFeatures;

                std::vector<int> spacepoint_ids_temp;
                for (int i=0; i < numSpacepoints; ++i){
                    spacepoint_ids_temp.push_back(i);
                }
                spacepoint_ids.push_back(spacepoint_ids_temp);
            }
        }
    } else if (fs::is_regular_file(filepath, ec)) {
        // read spacepoints table saved in csv
        std::vector<float> input_tensor_values_temp;
        processInput(input_file_path, input_tensor_values_temp);
        input_tensor_values.push_back(input_tensor_values_temp);

        int numSpacepoints = input_tensor_values_temp.size()/spacepointFeatures;

        std::vector<int> spacepoint_ids_temp;
        for (int i=0; i < numSpacepoints; ++i){
            spacepoint_ids_temp.push_back(i);
        }
        spacepoint_ids.push_back(spacepoint_ids_temp);
    }

    if (input_tensor_values.size() == 0) {
        std::cerr << "No input files found in " << input_file_path << std::endl;
        exit(EXIT_FAILURE);
    }

    // warmup the inferences
    for (int i=0; i < 10; ++i){
        ExaTrkXTime time;
        std::vector<std::vector<int> > track_candidates;
        infer->getTracks(input_tensor_values[0], spacepoint_ids[0], track_candidates, time);
    }

    // add a mutex to protect the following two variables
    int tot_evts = 0;
    std::mutex tot_evts_mutex;
    int tot_available_evts = input_tensor_values.size();
    std::cout << "Total number of available events: " << tot_available_evts << std::endl;

    auto run_one_file = [&](void) -> void {
        // measure the throughput
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        int niter = 0;
        while((end - start) < std::chrono::seconds(120)){
            std::vector<std::vector<int> > temp_candidates;
            ExaTrkXTime time;
            int idx = niter % tot_available_evts;
            infer->getTracks(input_tensor_values[idx], 
                             spacepoint_ids[idx], 
                             temp_candidates, time);
            ++niter;
            end = std::chrono::high_resolution_clock::now();
        }
        double infer_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000.;
        std::lock_guard<std::mutex> guard(tot_evts_mutex);
        tot_evts += niter;

        // // get thread id
        // std::stringstream ss;
        // ss << std::this_thread::get_id();
        // std::string thread_id = ss.str();

        // std::cout << "Thread: " << thread_id << " achieved throughput: " << niter / infer_time << " events per second." << std::endl;
        // std::cout << "\t(" << niter << " " << infer_time << ")" << std::endl;
    };

    auto loop_start_time = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(
        tbb::blocked_range<int>(0, nthreads),
        [&](tbb::blocked_range<int> r)
        {
            for (int i=r.begin(); i<r.end(); ++i) {
                run_one_file();
            }
        }
    );  // end parallel_for_each
    auto loop_end_time = std::chrono::high_resolution_clock::now();
    double tot_time = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count() / 1000.;
    std::cout << "Total throughput: " << tot_evts / tot_time << " events per second. (" << tot_evts << " " << tot_time << ")" << std::endl;
    return 0;
}
