#include "custom_node_api.hpp"

using namespace ovms;

void execute(
    const   TensorMap& in,
            TensorMap& out) {

    out["preprocessed_img"] = in.at("img_data");

    for (auto& [name, tensor] : out) {
        for (size_t i = 0; i < tensor.size(); i++)
            tensor[i] += 1;
    }
}
