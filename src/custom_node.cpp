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
#include <dlfcn.h>

#include <inference_engine.hpp>

#include "custom_node.hpp"

namespace ovms {

Status CustomNode::execute(ThreadSafeQueue<std::reference_wrapper<Node>>& notifyEndQueue) {
    spdlog::info("!!!!! CustomNode::execute");

    void* handle;
    if ((handle = dlopen(lib.c_str(), RTLD_LAZY)) == NULL) {
        spdlog::info("dlopen error");
        notifyEndQueue.push(*this);
        return StatusCode::OK;
    }

    execute_fn execute = (execute_fn) dlsym(handle, "execute");
    if (dlerror() != NULL) {
        spdlog::info("dlsym error");
        dlclose(handle);
        notifyEndQueue.push(*this);
        return StatusCode::OK;
    }

    spdlog::info("execute prepared");

    // prepare input

    // convert from blob map to api supported format
    TensorMap inputs;
    
    for (const auto& [name, blob] : this->inputBlobs) {
        spdlog::info("converting {}", name);
        Tensor tensor(blob->byteSize());
        std::memcpy(tensor.data(), (void*)blob->buffer(), blob->byteSize());
        inputs[name] = std::move(tensor);
    }

    execute(inputs, outputs);

    spdlog::info("executed");    

    dlclose(handle);
    notifyEndQueue.push(*this);
    return StatusCode::OK;
}

Status CustomNode::fetchResults(BlobMap& outputs) {
    spdlog::info("!!!!! CustomNode::fetchResults");
    // convert from api format to blob map

    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& output_name = pair.first;
            if (outputs.count(output_name) == 1) {
                continue;
            }

            try {
                InferenceEngine::TensorDesc description;
                description.setPrecision(InferenceEngine::Precision::FP32);
                description.setDims({1, 3, 224, 224});
                spdlog::info("output_name:{}/{}", this->nodeOutputNameAlias.at(output_name), output_name);
                Tensor& tensor = this->outputs.at(this->nodeOutputNameAlias.at(output_name));
                auto blob = InferenceEngine::make_shared_blob<float>(description, (float*)tensor.data());
                outputs.emplace(std::make_pair(output_name, std::move(blob)));
            } catch (const std::out_of_range& e) {
                Status status = StatusCode::OV_INTERNAL_SERIALIZATION_ERROR;
                SPDLOG_DEBUG("[Node: {}] Error during getting tensor {}; exception message: {}", getName(), status.string(), e.what());
                return status;
            }
            spdlog::debug("[Node: {}]: Blob with name {} has been prepared", getName(), output_name);
        }
    }

    return StatusCode::OK;
}

}  // namespace ovms
