//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#pragma once

#include <string>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include "node.hpp"
#include "custom_nodes/custom_node_api.hpp"

namespace ovms {

class CustomNode : public Node {
    const std::unordered_map<std::string, std::string> nodeOutputNameAlias;
    std::string lib;

    TensorMap outputs;
public:
    CustomNode(const std::string& nodeName, const std::string& lib, std::unordered_map<std::string, std::string> nodeOutputNameAlias = {}) :
        Node(nodeName), lib(lib), nodeOutputNameAlias(nodeOutputNameAlias) {
            for (const auto& [v1, v2] : this->nodeOutputNameAlias) {
                spdlog::info("XXXX---- {}, {}, {}", getName(), v1, v2);
            }
        }

    Status execute(ThreadSafeQueue<std::reference_wrapper<Node>>& notifyEndQueue) override;
    Status fetchResults(BlobMap& outputs) override;
};

}  // namespace ovms
