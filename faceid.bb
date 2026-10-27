SUMMARY = "NATS-based face recognition service and client"
DESCRIPTION = "Face recognition service with LiteRT inference and Redis, Qdrant or in-memory vector storage."
HOMEPAGE = "https://code.hidglobal.com/nova/nova-maximus-services-face-recognition"

# No application licence is supplied in this repository.
LICENSE = "CLOSED"



# Replace with your actual repository URI or local source folder
SRC_URI = "git://git@github.com/firehills/imx-faceid.git;protocol=https;branch=main"
# SRC_URI = "git://git@code.hidglobal.com/nova/nova-maximus-services-face-recognition.git;protocol=ssh;branch=master"
SRCREV = "${AUTOREV}"

#S = "${WORKDIR}/git"


# 1. Inherit the PEP-517 build class
inherit python_pep517

# Build-time host requirements
DEPENDS += " \
    python3-setuptools-native \
"

# Runtime target dependencies mapped directly from your [project.dependencies]
RDEPENDS:${PN} += " \
    python3-core \
    python3-datetime \
    python3-flatbuffers \
    python3-json \
    python3-logging \
    python3-numpy \
    python3-pillow \
    python3-pyyaml \
"

# Optional Dependencies (Yocto Package Configurations)
# Maps your project's optional features to Yocto package flags
PACKAGECONFIG ??= " "
PACKAGECONFIG[onnx] = ",,,python3-onnxruntime"
PACKAGECONFIG[litert] = ",,,python3-ai-edge-litert"

# Dynamically apply PACKAGECONFIG optional runtime dependencies
RDEPENDS:${PN} += "${@bb.utils.contains('PACKAGECONFIG', 'onnx', '${PACKAGECONFIG[onnx]}', '', d)}"
RDEPENDS:${PN} += "${@bb.utils.contains('PACKAGECONFIG', 'litert', '${PACKAGECONFIG[onnx]}', '', d)}"
