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
#include "localfilesystem.hpp"

#include <filesystem>
#include <fstream>
#include <string>

#include <spdlog/spdlog.h>

#if defined(__APPLE__) || defined(__NetBSD__)
#define st_mtim st_mtimespec
#endif

namespace ovms {

namespace fs = std::filesystem;
constexpr uint64_t NANOS_PER_SECOND = 1000000000;

StatusCode LocalFileSystem::fileExists(const std::string& path, bool* exists) {
    try {
        *exists = fs::exists(path);
    } catch (fs::filesystem_error& e) {
        SPDLOG_DEBUG("Couldn't access path {}", e.what());
        return StatusCode::PATH_INVALID;
    }

    return StatusCode::OK;
}

StatusCode LocalFileSystem::isDirectory(const std::string& path, bool* is_dir) {
    try {
        *is_dir = fs::is_directory(path);
    } catch (fs::filesystem_error& e) {
        SPDLOG_DEBUG("Couldn't access path {}", e.what());
        return StatusCode::PATH_INVALID;
    }

    return StatusCode::OK;
}

StatusCode LocalFileSystem::fileModificationTime(const std::string& path, int64_t* mtime_ns) {
    // For last modification time we'll be using c style
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        SPDLOG_DEBUG("Couldn't access path {}", path);
        return StatusCode::PATH_INVALID;
    }

    *mtime_ns = st.st_mtim.tv_sec * NANOS_PER_SECOND + st.st_mtim.tv_nsec;

    return StatusCode::OK;
}

StatusCode LocalFileSystem::getDirectoryContents(const std::string& path, files_list_t* contents) {
    try {
        for (const auto& entry : fs::directory_iterator(path)) {
            contents->insert(entry.path().string());
        }
    } catch (fs::filesystem_error& e) {
        SPDLOG_DEBUG("Couldn't access path {}", e.what());
        return StatusCode::PATH_INVALID;
    }

    return StatusCode::OK;
}

StatusCode LocalFileSystem::getDirectorySubdirs(const std::string& path, files_list_t* subdirs) {
    try {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory()) {
                subdirs->insert(entry.path().string());
            }
        }
    } catch (fs::filesystem_error& e) {
        SPDLOG_DEBUG("Couldn't access path {}", e.what());
        return StatusCode::PATH_INVALID;
    }

    return StatusCode::OK;
}

StatusCode LocalFileSystem::getDirectoryFiles(const std::string& path, files_list_t* files) {
    try {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (!entry.is_directory()) {
                files->insert(entry.path().string());
            }
        }
    } catch (fs::filesystem_error& e) {
        SPDLOG_DEBUG("Couldn't access path {}", e.what());
        return StatusCode::PATH_INVALID;
    }

    return StatusCode::OK;
}

StatusCode LocalFileSystem::readTextFile(const std::string& path, std::string* contents) {
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input) {
        SPDLOG_DEBUG("Couldn't access path {}", path);
        return StatusCode::PATH_INVALID;
    }

    input.seekg(0, std::ios::end);
    contents->resize(input.tellg());
    input.seekg(0, std::ios::beg);
    input.read(&(*contents)[0], contents->size());
    input.close();

    return StatusCode::OK;
}

StatusCode LocalFileSystem::downloadFileFolder(const std::string& path, std::string* local_path) {
    // For LocalFileSystem there is no need to download
    *local_path = path;
    return StatusCode::OK;
}

StatusCode LocalFileSystem::deleteFileFolder(const std::string& path) {
    // Since we're using an orignal folder, do nothing
    return StatusCode::OK;
}

}  // namespace ovms