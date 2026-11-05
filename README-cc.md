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
Use `level: debug` for request/response details (formerly `verbose`), `info`
for service status, `error` for errors only, and `off` to suppress service logs.
Fatal startup errors are always reported on stderr, even when logging is off.
Command-line help remains plain stdout, not a log message.

Python logging, dependencies, and configuration are unchanged. This config
controls application logs, not third-party libraries' own internal diagnostics.
