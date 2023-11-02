#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <filesystem>
#include <unistd.h>
#include <limits.h>

// #include <tbb/parallel_for_each.h>
// #include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/info.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <unistd.h>

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
                fprintf(stderr, "Usage: %s [-hv] [-d input_file_path] [-m model_path] [-t number of threads]\n", argv[0]);
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
    // tbb::task_scheduler_init init(nthreads);    
    // int num_threads = tbb::info::default_concurrency();
    // if (nthreads > num_threads) {
    //     std::cerr << "Warning: number of threads requested (" << nthreads \
    //     << ") is greater than the number of threads available (" << num_threads << ")." << std::endl;
    // }
    tbb::task_arena arena(nthreads);

    std::cout << "Input file: " << input_file_path << std::endl;

    std::unique_ptr<ExaTrkXTrackFinding> infer;
    ExaTrkXTrackFinding::Config config{model_path, verbose};
    infer = std::make_unique<ExaTrkXTrackFinding>(config);
    // std::cout << "Running Inference with local CPUs" << std::endl;

    const fs::path filepath(input_file_path);
    std::error_code ec;
    ExaTrkXTimeList tot_time;
    ExaTrkXTimer timer;
    timer.start();
    int tot_tracks = 0;
    int ievt = 0;

    auto run_one_file = [&](const fs::path& in_file_name) -> void {
        // read spacepoints table saved in csv
        std::vector<float> input_tensor_values;
        processInput(in_file_name, input_tensor_values);
        int64_t spacepointFeatures = 3;

        int numSpacepoints = input_tensor_values.size()/spacepointFeatures;

        std::vector<int> spacepoint_ids;
        for (int i=0; i < numSpacepoints; ++i){
            spacepoint_ids.push_back(i);
        }
        std::vector<std::vector<int> > track_candidates;
        ExaTrkXTime time;
        infer->getTracks(input_tensor_values, spacepoint_ids, track_candidates, time);
        tot_time.add(time);
        tot_tracks += track_candidates.size();

        // dumpTrackCandidate(track_candidates);
    };


    if (fs::is_directory(filepath, ec)) {
        if (nthreads > 1) {
            // concurrent execution of all files in directory
            std::vector<std::string>  filenames;
            for (auto& entry : fs::directory_iterator(filepath)) {
                if (fs::is_regular_file(entry.path())) {
                    filenames.push_back(entry.path().string());
                }
            }
            int nfiles = std::distance(filenames.begin(), filenames.end());
            // std::cout << "Running " << nfiles << " files in " << nthreads << " threads." << std::endl;

            // // set affinity for each thread
            // cpu_set_t cpuset;
            // CPU_ZERO(&cpuset);
            // int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
            // int ncpus_per_thread = ncpus/nthreads;
            // for (int i=0; i < nthreads; ++i) {
            //     for (int j=0; j < ncpus_per_thread; ++j) {
            //         CPU_SET(i*ncpus_per_thread + j, &cpuset);
            //     }
            //     pthread_t thread = pthread_self();
            //     pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
            // }

            // // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/Cook_Until_Done_parallel_do.html
            // // the fetching of work is serial
            // tbb::parallel_for_each(
            //     filenames.begin(), filenames.end(),
            //     [&](const std::string& fname) {
            //         run_one_file(fname);
            //     });  // end parallel_for_each

            // use tbb parallel_for to process each file
            arena.execute([&](){
                tbb::parallel_for(
                    tbb::blocked_range<int>(0, nfiles),
                    [&](const tbb::blocked_range<int>& r) {
                        for (int i=r.begin(); i < r.end(); ++i) {
                            run_one_file(filenames[i]);
                        }
                    });  // end parallel_for
            });

        } else {
            // sequential execution of all files in directory
            for(auto& entry : fs::directory_iterator(filepath)) {
                if (fs::is_regular_file(entry.path())) {
                    // std::cout << "Processing file: " << entry.path().string() << std::endl;
                    run_one_file(entry.path().string());
                }
            }
        }

    } else if (fs::is_regular_file(filepath, ec)) {
        run_one_file(filepath);
    } else {
        std::cerr << "Error: " << filepath << " is not a file or directory." << std::endl;
        exit(EXIT_FAILURE);
    }
    printf("Total time: %.4f seconds\n", timer.stopAndGetElapsedTime());


    // printf("Total %d tracks in %d events.\n", tot_tracks, tot_time.numEvts());
    // tot_time.summary();
    // printf("-----------------------------------------------------\n");
    // printf("Summary of the first event\n");
    // tot_time.summaryOneEvent(0);
    // printf("-----------------------------------------------------\n");
    // printf("Summary of without first 1 event\n");
    // tot_time.summary(1);
    // printf("Summary of the last event\n");
    // tot_time.summaryOneEvent(tot_time.numEvts()-1);
    // std::stringstream ss;
    // ss << "time_t" << nthreads << "_s" << server_type << ".csv";
    // tot_time.save(ss.str());
    return 0;
}
