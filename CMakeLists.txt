cmake_minimum_required(VERSION 3.21)

project(imx_faceid_cpp VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

find_package(spdlog 1.17.0 EXACT CONFIG QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(spdlog
        URL https://codeload.github.com/gabime/spdlog/tar.gz/refs/tags/v1.17.0
        URL_HASH SHA256=d8862955c6d74e5846b3f580b1605d2428b11d97a410d86e2fb13e857cd3a744
    )
    FetchContent_MakeAvailable(spdlog)
endif()

find_program(FLATC_EXECUTABLE flatc REQUIRED)
find_path(FLATBUFFERS_INCLUDE_DIR flatbuffers/flatbuffers.h REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)
pkg_check_modules(ONNXRUNTIME REQUIRED IMPORTED_TARGET libonnxruntime)
pkg_check_modules(TFLITE REQUIRED IMPORTED_TARGET tensorflow2-lite)
pkg_check_modules(YAMLCPP REQUIRED IMPORTED_TARGET yaml-cpp)
pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
pkg_check_modules(NATS REQUIRED IMPORTED_TARGET libnats)

find_library(OPENCV_CORE_LIBRARY NAMES opencv_core
    HINTS ${OPENCV_LIBRARY_DIRS} REQUIRED)
find_library(OPENCV_IMGPROC_LIBRARY NAMES opencv_imgproc
    HINTS ${OPENCV_LIBRARY_DIRS} REQUIRED)
find_library(OPENCV_IMGCODECS_LIBRARY NAMES opencv_imgcodecs
    HINTS ${OPENCV_LIBRARY_DIRS} REQUIRED)

find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
    set(GRPC_CPP_PLUGIN_TARGET "$ENV{SDKTARGETSYSROOT}/usr/bin/grpc_cpp_plugin")
    if(CMAKE_CROSSCOMPILING AND EXISTS "${GRPC_CPP_PLUGIN_TARGET}")
        find_program(QEMU_AARCH64_EXECUTABLE qemu-aarch64 REQUIRED)
        set(GRPC_CPP_PLUGIN_WRAPPER
            "${CMAKE_CURRENT_BINARY_DIR}/grpc_cpp_plugin_wrapper")
        file(WRITE "${GRPC_CPP_PLUGIN_WRAPPER}"
            "#!/bin/sh\n"
            "exec \"${QEMU_AARCH64_EXECUTABLE}\" -L \"$ENV{SDKTARGETSYSROOT}\" \"${GRPC_CPP_PLUGIN_TARGET}\" \"$@\"\n"
        )
        file(CHMOD "${GRPC_CPP_PLUGIN_WRAPPER}"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
        set(GRPC_CPP_PLUGIN_EXECUTABLE "${GRPC_CPP_PLUGIN_WRAPPER}")
        message(STATUS
            "Using target grpc_cpp_plugin through qemu-aarch64: ${GRPC_CPP_PLUGIN_TARGET}")
    else()
        message(FATAL_ERROR
            "Could not find host grpc_cpp_plugin or a target plugin at ${GRPC_CPP_PLUGIN_TARGET}")
    endif()
endif()
find_path(PROTOBUF_PROTO_DIR google/protobuf/timestamp.proto
    HINTS
        "$ENV{OECORE_NATIVE_SYSROOT}/usr/include"
        "$ENV{SDKTARGETSYSROOT}/usr/include"
        ${PROTOBUF_INCLUDE_DIRS}
    REQUIRED
)

set(FACEID_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/schemas/faceid.fbs")
set(FACEID_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(FACEID_GENERATED_HEADER "${FACEID_GENERATED_DIR}/faceid_generated.h")

add_custom_command(
    OUTPUT "${FACEID_GENERATED_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${FACEID_GENERATED_DIR}"
    COMMAND "${FLATC_EXECUTABLE}" --cpp --scoped-enums --gen-object-api
            -o "${FACEID_GENERATED_DIR}" "${FACEID_SCHEMA}"
    DEPENDS "${FACEID_SCHEMA}"
    COMMENT "Generating C++ FlatBuffers bindings for faceid.fbs"
    VERBATIM
)

add_custom_target(faceid_schema DEPENDS "${FACEID_GENERATED_HEADER}")

add_library(faceid_codec STATIC
    cpp/src/flatbuffer_codec.cpp
    "${FACEID_GENERATED_HEADER}"
)
add_dependencies(faceid_codec faceid_schema)
target_include_directories(faceid_codec PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
    "${FACEID_GENERATED_DIR}"
    "${FLATBUFFERS_INCLUDE_DIR}"
)

add_library(faceid_config STATIC cpp/src/config.cpp)
target_include_directories(faceid_config PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
)
target_link_libraries(faceid_config PUBLIC PkgConfig::YAMLCPP)

add_library(faceid_logging STATIC cpp/src/logging.cpp)
target_include_directories(faceid_logging PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
)
target_link_libraries(faceid_logging PUBLIC spdlog::spdlog PkgConfig::YAMLCPP)
configure_file(spdlog.yaml spdlog.yaml COPYONLY)

add_library(faceid_inference STATIC cpp/src/inference.cpp)
target_include_directories(faceid_inference PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
    ${OPENCV_INCLUDE_DIRS}
)
target_link_libraries(faceid_inference PUBLIC
    faceid_config
    faceid_codec
    ${OPENCV_IMGCODECS_LIBRARY}
    ${OPENCV_IMGPROC_LIBRARY}
    ${OPENCV_CORE_LIBRARY}
    PkgConfig::ONNXRUNTIME
    PkgConfig::TFLITE
)

set(QDRANT_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/qdrant")
set(QDRANT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/qdrant")
set(QDRANT_PROTO_FILES
    "${QDRANT_PROTO_DIR}/collections.proto"
    "${QDRANT_PROTO_DIR}/collections_service.proto"
    "${QDRANT_PROTO_DIR}/json_with_int.proto"
    "${QDRANT_PROTO_DIR}/points.proto"
    "${QDRANT_PROTO_DIR}/points_service.proto"
    "${QDRANT_PROTO_DIR}/qdrant_common.proto"
    "${QDRANT_PROTO_DIR}/qdrant_health.proto"
)
set(QDRANT_GRPC_PROTO_FILES
    "${QDRANT_PROTO_DIR}/collections_service.proto"
    "${QDRANT_PROTO_DIR}/points_service.proto"
    "${QDRANT_PROTO_DIR}/qdrant_health.proto"
)
set(QDRANT_PROTO_GENERATED_SOURCES)
set(QDRANT_PROTO_GENERATED_HEADERS)
set(QDRANT_GRPC_GENERATED_SOURCES)
set(QDRANT_GRPC_GENERATED_HEADERS)
foreach(PROTO_FILE IN LISTS QDRANT_PROTO_FILES)
    get_filename_component(PROTO_NAME "${PROTO_FILE}" NAME_WE)
    list(APPEND QDRANT_PROTO_GENERATED_SOURCES
        "${QDRANT_GENERATED_DIR}/${PROTO_NAME}.pb.cc"
    )
    list(APPEND QDRANT_PROTO_GENERATED_HEADERS
        "${QDRANT_GENERATED_DIR}/${PROTO_NAME}.pb.h"
    )
endforeach()
foreach(PROTO_FILE IN LISTS QDRANT_GRPC_PROTO_FILES)
    get_filename_component(PROTO_NAME "${PROTO_FILE}" NAME_WE)
    list(APPEND QDRANT_GRPC_GENERATED_SOURCES
        "${QDRANT_GENERATED_DIR}/${PROTO_NAME}.grpc.pb.cc"
    )
    list(APPEND QDRANT_GRPC_GENERATED_HEADERS
        "${QDRANT_GENERATED_DIR}/${PROTO_NAME}.grpc.pb.h"
    )
endforeach()

add_custom_command(
    OUTPUT
        ${QDRANT_PROTO_GENERATED_SOURCES}
        ${QDRANT_PROTO_GENERATED_HEADERS}
        ${QDRANT_GRPC_GENERATED_SOURCES}
        ${QDRANT_GRPC_GENERATED_HEADERS}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${QDRANT_GENERATED_DIR}"
    COMMAND "${PROTOC_EXECUTABLE}"
            "--proto_path=${QDRANT_PROTO_DIR}"
            "--proto_path=${PROTOBUF_PROTO_DIR}"
            "--cpp_out=${QDRANT_GENERATED_DIR}"
            ${QDRANT_PROTO_FILES}
    COMMAND "${PROTOC_EXECUTABLE}"
            "--proto_path=${QDRANT_PROTO_DIR}"
            "--proto_path=${PROTOBUF_PROTO_DIR}"
            "--grpc_out=${QDRANT_GENERATED_DIR}"
            "--plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}"
            ${QDRANT_GRPC_PROTO_FILES}
    DEPENDS ${QDRANT_PROTO_FILES}
    COMMENT "Generating Qdrant v1.19.2 C++ gRPC bindings"
    VERBATIM
)

add_library(faceid_qdrant_proto STATIC
    ${QDRANT_PROTO_GENERATED_SOURCES}
    ${QDRANT_PROTO_GENERATED_HEADERS}
    ${QDRANT_GRPC_GENERATED_SOURCES}
    ${QDRANT_GRPC_GENERATED_HEADERS}
)
target_include_directories(faceid_qdrant_proto PUBLIC "${QDRANT_GENERATED_DIR}")
target_link_libraries(faceid_qdrant_proto PUBLIC
    PkgConfig::PROTOBUF
    PkgConfig::GRPCPP
)

add_library(faceid_score STATIC cpp/src/vector_store_score.cpp)
target_include_directories(faceid_score PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
)

add_library(faceid_vector_store STATIC cpp/src/vector_store.cpp)
target_include_directories(faceid_vector_store PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
)
target_link_libraries(faceid_vector_store PUBLIC
    faceid_codec
    faceid_qdrant_proto
    faceid_score
)

add_executable(imx_faceid_cpp
    cpp/src/main.cpp
    cpp/src/service.cpp
)
target_compile_definitions(imx_faceid_cpp PRIVATE
    IMX_FACEID_CPP_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(imx_faceid_cpp PRIVATE
    faceid_logging
    faceid_inference
    faceid_vector_store
    PkgConfig::NATS
)

if(BUILD_TESTING)
    add_executable(test_logging cpp/tests/test_logging.cpp)
    target_link_libraries(test_logging PRIVATE faceid_logging)
    add_test(NAME logging COMMAND test_logging "${CMAKE_CURRENT_SOURCE_DIR}/spdlog.yaml")
    add_test(NAME cpp_help COMMAND imx_faceid_cpp --help)
    set_tests_properties(cpp_help PROPERTIES
        PASS_REGULAR_EXPRESSION "logging-config"
    )
    add_test(NAME cpp_startup_version
        COMMAND "${CMAKE_COMMAND}"
            "-DFACEID_EXECUTABLE=$<TARGET_FILE:imx_faceid_cpp>"
            "-DFACEID_EMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR}"
            "-DFACEID_VERSION=${PROJECT_VERSION}"
            "-DFACEID_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/test_startup_version.cmake"
    )
    add_executable(test_vector_store_scores cpp/tests/test_vector_store_scores.cpp)
    target_link_libraries(test_vector_store_scores PRIVATE faceid_score)
    add_test(NAME vector_store_scores COMMAND test_vector_store_scores)
endif()