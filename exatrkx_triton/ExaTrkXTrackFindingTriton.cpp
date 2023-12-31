#include "ExaTrkXTrackFindingTriton.hpp"

#include "grpc_client.h"
#include "grpc_service.pb.h"
namespace tc = triton::client;


ExaTrkXTrackFindingTriton::ExaTrkXTrackFindingTriton(
    const ExaTrkXTrackFindingTriton::Config& config): m_cfg(config)
{
    bool verbose = false;
    uint32_t client_timeout = 0;
    std::string model_version = "";
    m_client = std::make_unique<ExaTrkXTriton>(m_cfg.modelName, m_cfg.url, model_version, client_timeout, verbose);
}


// The main function that runs the Exa.TrkX pipeline
// Be care of sharpe corners.
void ExaTrkXTrackFindingTriton::getTracks(
    std::vector<float>& inputValues,
    std::vector<int>& spacepointIDs,
    std::vector<std::vector<int> >& trackCandidates) const
{

    // ************
    // TrackFinding
    // ************
    int64_t numSpacepoints = inputValues.size() / m_cfg.spacepointFeatures;
    std::vector<int64_t> embedInputShape{numSpacepoints, m_cfg.spacepointFeatures};


    m_client->ClearInput();
    m_client->AddInput<float>("FEATURES", embedInputShape, inputValues);
    std::vector<int64_t> trackLabels;
    std::vector<int64_t> trackLabelsShape{numSpacepoints, 1};
    m_client->GetOutput<int64_t>("LABELS", trackLabels, trackLabelsShape);


    int idx = 0;
    // std::cout << "size of components: " << trackLabels.size() << std::endl;
    if (trackLabels.size() == 0)  return;
    trackCandidates.clear();

    int existTrkIdx = 0;
    // map labeling from MCC to customized track id.
    std::map<int32_t, int32_t> trackLableToIds;

    for(int32_t idx=0; idx < numSpacepoints; ++idx) {
        int32_t trackLabel = trackLabels[idx];
        int spacepointID = spacepointIDs[idx];

        int trkId;
        if(trackLableToIds.find(trackLabel) != trackLableToIds.end()) {
            trkId = trackLableToIds[trackLabel];
            trackCandidates[trkId].push_back(spacepointID);
        } else {
            // a new track, assign the track id
            // and create a vector
            trkId = existTrkIdx;
            trackCandidates.push_back(std::vector<int>{spacepointID});
            trackLableToIds[trackLabel] = trkId;
            existTrkIdx++;
        }
    }
}
