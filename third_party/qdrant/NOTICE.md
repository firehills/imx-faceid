# Qdrant gRPC definitions

The protobuf files in this directory are from the Qdrant `v1.19.2` release,
under `lib/api/src/grpc/proto`. They are used to generate the C++ client stubs
for the collections and points services. The upstream Apache-2.0 license is in
`LICENSE`.

`qdrant_health.proto` is a minimal extraction of the `Qdrant.HealthCheck`
service and its request/reply messages from upstream `qdrant.proto`. The
service name, method name, package, and field numbers match that release. It
avoids importing unrelated Qdrant services into this client.