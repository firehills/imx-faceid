# C++ service

## Build and test

```sh
source /home/phill/repo/yocto-sdk/environment-setup-armv8-2a-poky-linux
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CROSSCOMPILING_EMULATOR="qemu-aarch64;-L;$SDKTARGETSYSROOT" \
  -DFLATBUFFERS_INCLUDE_DIR="$SDKTARGETSYSROOT/usr/include"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The SDK dependencies are described in [AGENTS.md](AGENTS.md).
The C++ build requires **spdlog 1.17.0 exactly**. CMake uses an installed
matching package or downloads the SHA-256-verified release. For offline builds,
provide an unpacked 1.17.0 release with
`-DFETCHCONTENT_SOURCE_DIR_SPDLOG=/path/to/spdlog-1.17.0`.

## Logging

```sh
./build/imx_faceid_cpp --config config.yaml --logging-config spdlog.yaml
```

Both paths default to the shown filenames in the working directory. CMake also
copies [spdlog.yaml](spdlog.yaml) into the build directory. Deploy this file
alongside the executable, or pass its absolute path.

## Stopping

Ctrl+C (SIGINT) or SIGTERM stops only `imx_faceid_cpp`, never the NATS server.
If NATS is connected, the service drains in-flight requests (up to 30s). If NATS
is disconnected, it exits immediately. A second Ctrl+C forces exit with code 130.

The C++ service configures spdlog from this separate YAML file at startup:

- `level`: `trace`, `debug`, `info`, `warn`, `error`, `critical`, or `off`.
- `pattern`: non-empty spdlog formatting pattern.
- `console`: enable thread-safe stderr output (color when supported).
- `file`: optional append-only log file; relative paths resolve from the
  logging configuration's directory. File logging does not rotate automatically.

`level`, `pattern`, and `console` are required. At least one output must be
configured, even with `level: off`. Invalid or missing configuration fails startup
with a diagnostic on stderr. Errors flush immediately; normal shutdown flushes
all pending output.

The shared application YAML's legacy `debug` setting is **ignored by C++**.
Use `level: debug` for request/response subjects and sizes (formerly `verbose`),
`trace` to also log each decoded NATS request and encoded response (IDs, names,
`reply_to`, thresholds, timestamps, image dimensions/encoding, candidate matches,
and byte/record counts; image bytes, embeddings, and `custom` contents are never
printed), `info`
for service status and per-request inference timing, `error` for errors only,
and `off` to suppress service logs.

At `info`, each enrolment and identity query logs one line, for example:

```
Inference query track_id=track-7 runtime=onnx preprocess_ms=0.64 inference_ms=12.31
```

`preprocess_ms` covers image decoding, alignment, resizing, and tensor conversion.
`inference_ms` covers ONNX Runtime or LiteRT execution plus embedding
normalization. Neither includes time spent waiting for another request's
inference to finish, or Qdrant access.
Fatal startup errors are always reported on stderr, even when logging is off.
Command-line help remains plain stdout, not a log message.

Python logging, dependencies, and configuration are unchanged. This config
controls application logs, not third-party libraries' own internal diagnostics.
