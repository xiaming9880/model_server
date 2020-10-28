#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace ovms {

typedef std::vector<std::uint8_t> Tensor;
typedef std::unordered_map<std::string, Tensor> TensorMap;

typedef void (*execute_fn)(const TensorMap&, TensorMap&);

}  // namespace ovms

extern "C"
void execute(
    const   ovms::TensorMap& in,
            ovms::TensorMap& out);
