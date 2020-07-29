# Set the Custom Extension Library

When the graph include layers and operations aren't supported by the device plug-in, you might want to extend the Inference Engine. To do this, create a custom kernel for network layers. You can implement these custom layers in the  OpenVINO&trade; Model Server to handle these inference requests.

See [docs.openvinotoolkit.org](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_your_kernels_into_IE.html) to learn to create this extension.

Compile the Inference Engine extension as a separate library and copy it to the OpenVINO model server. The OpenVINO model server looks for the extension library in the path that you define in environment variable `CPU_EXTENSION`. Without this variable, a [standard set of layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)
 is supported.

The Docker image that contains the OpenVINO model server doesn't include all of the tools and sub-components to compile the extension library; you might need to execute this process on a separate host. After compiling the CPU extension, attach it to the Docker container with OpenVINO model server and reference it by setting its path similar to:

```bash
docker run --rm -d -v /models/:/opt/ml:ro -p 9001:9001 --env CPU_EXTENSION=/opt/ml/libcpu_extension.so  ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
```  
