#include "ExaTrkXUtils.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

void buildEdges(
    std::vector<float>& embedFeatures,
    std::vector<int64_t>& edgeList,
    int64_t numSpacepoints,
    int embeddingDim,    // dimension of embedding space
    float rVal, // radius of the ball
    int kVal    // number of nearest neighbors
) {

    // build the index without training
    float radius = rVal * rVal;
    faiss::IndexFlatL2 index(embeddingDim);
    // printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(numSpacepoints, embedFeatures.data());

    faiss::idx_t* I = new faiss::idx_t[kVal * numSpacepoints];
    float* D = new float[kVal * numSpacepoints];
    index.search(numSpacepoints, embedFeatures.data(), kVal, D, I);

    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    // only keep edges that are within the radius of the ball
    for (int64_t i = 0; i < numSpacepoints; ++i) {
        for (int64_t j = 1; j < kVal; ++j) {
            if (D[i * kVal + j] <= radius) {
                rows.push_back(i);
                cols.push_back(I[i * kVal + j]);
            }
        }
    }
    edgeList.resize(rows.size() + cols.size());
    std::copy(rows.begin(), rows.end(), edgeList.begin());
    std::copy(cols.begin(), cols.end(), edgeList.begin() + rows.size());

    delete [] I;
    delete [] D;
}
