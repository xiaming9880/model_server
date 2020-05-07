#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import os
from src.api.models.model import Model, LabelsFileLoadingError, LabelsFileContentError, LabelsFileFormatError
from src.api.models.vehicle_detection_adas_model import VehicleDetectionAdas
from json import JSONDecodeError

IMAGES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),'labels_files')

adas_json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../../../ams_models/vehicle_detection_adas_model.json")
bad_json_path = os.path.join(IMAGES_DIR,"bad_json.json")
bad_path = os.path.join(IMAGES_DIR,"bad_path.json")
bad_labels_path = os.path.join(IMAGES_DIR,"bad_labels.json")


def test_model_object():
    test_model = VehicleDetectionAdas("test-model","ovms-connector",adas_json_path)
    assert test_model.model_name == "test-model"
    assert test_model.ovms_connector == "ovms-connector"


def test_bad_json_loading():
    with pytest.raises(JSONDecodeError):
         test_model = VehicleDetectionAdas("test-model","ovms-connector",bad_json_path)


def test_bad_path_loading():
    with pytest.raises(FileNotFoundError):
         test_model = VehicleDetectionAdas("test-model","ovms-connector", bad_path)
        

def test_bad_format_loading():
    with pytest.raises(KeyError):
         test_model = VehicleDetectionAdas("test-model","ovms-connector",bad_labels_path)

