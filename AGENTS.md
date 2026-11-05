# C++ Service Guide

## Project Layout

- `schemas/faceid.fbs` is the NATS request/response contract. CMake generates
  its C++ bindings into the build directory.
- `cpp/include/faceid/` contains the documented public C++ APIs.
- `cpp/src/` contains configuration loading, FlatBuffer codecs, ONNX/LiteRT
  inference, the Qdrant gRPC store, the NATS service, and the executable.
- `cpp/tests/` contains C++ tests registered with CTest.
- `third_party/qdrant/` contains the Qdrant v1.19.2 protobuf definitions used
  for C++ gRPC code generation. Its upstream license and source notes are kept
  alongside the definitions.
- `src/imx_faceid/` remains the Python implementation; the C++ service is
  built as a separate executable.

## SDK Requirements

Build the C++ service with the aarch64 Yocto SDK. The target sysroot must provide
OpenCV, ONNX Runtime, TensorFlow Lite, yaml-cpp, libnats, FlatBuffers headers,
protobuf, and gRPC++. The host SDK tools must provide `flatc`, `protoc`, and
`grpc_cpp_plugin`. The host protobuf compiler and target protobuf runtime must
be compatible; the gRPC plugin and target gRPC++ library must also match.

The current SDK has protobuf 33.6, libnats 3.9.3, FlatBuffers target headers,
and the `grpc++` target package. It does not provide a host `grpc_cpp_plugin`,
but it includes the target plugin; when cross-compiling, CMake wraps that plugin
with `qemu-aarch64`. CMake requires pkg-config modules named `protobuf`,
`grpc++`, and `libnats`.

## Build And Test

Source the SDK setup script, then configure, build, and run CTest:

```sh
source /home/phill/repo/yocto-sdk/environment-setup-armv8-2a-poky-linux
cmake -S . -B /tmp/imx-faceid-cpp-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CROSSCOMPILING_EMULATOR="qemu-aarch64;-L;$SDKTARGETSYSROOT" \
  -DFLATBUFFERS_INCLUDE_DIR="$SDKTARGETSYSROOT/usr/include"
cmake --build /tmp/imx-faceid-cpp-build --parallel
ctest --test-dir /tmp/imx-faceid-cpp-build --output-on-failure
```

The build produces `imx_faceid_cpp`. Run it with a YAML configuration file:

```sh
/tmp/imx-faceid-cpp-build/imx_faceid_cpp --config config.yaml
```

The C++ client uses Qdrant gRPC; unlike the Python client, it does not fall
back to REST. Collection vector size, vector type, and distance metric are
validated at startup. CMake generates FlatBuffers and Qdrant protobuf sources
under the selected build directory; do not edit generated files there.

The Python test suite remains separate and can be run with:

```sh
python -m pytest
```