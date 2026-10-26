# imx-faceid

A Python face-identification service that receives FlatBuffer messages over NATS,
runs an ONNX or LiteRT embedding model, and stores/searches vectors through
Qdrant.

## Python and virtual environment

Use Python 3.14.5 and create a project-local virtual environment:

```sh
python3.14 -m venv .venv
.venv/bin/python --version # Must report Python 3.14.5 or newer.
.venv/bin/python -m pip install --upgrade pip
```

Install the runtime for the model format in use:

```sh
.venv/bin/python -m pip install -e '.[onnx]'
# Or:
.venv/bin/python -m pip install -e '.[litert]'
```

Install both extras if the deployed configuration may select either runtime.
The ONNX adapter uses the CPU provider. LiteRT uses CPU unless
`model.delegate_library` is configured. Delegate options are passed to LiteRT;
delegate initialization errors are reported rather than silently falling back.

Copy [config.example.yaml](./config.example.yaml) to `config.yaml`, set the
model-specific preprocessing values, and run:

```sh
.venv/bin/imx-faceid --config config.yaml
```

The project metadata requires Python 3.14.5 or newer. The venv must use a
matching Python installation.

## Configuration

The example config includes editable NATS subjects, NATS URL/authentication,
Qdrant host and HTTP/gRPC ports, optional Qdrant API key, collection names,
distance metric, model path/runtime, tensor layout and names, preprocessing,
alignment, optional LiteRT delegate settings, and a top-level `debug` mode.

`debug` controls logging verbosity and accepts:

* `none`: disable service logging output
* `error`: emit errors only
* `info`: emit normal operational logs (default)
* `verbose`: include debug-level logs with decoded key fields from incoming
  requests and outgoing response FlatBuffer messages (image bytes are not
  logged)

`model.path`, `input_layout`, `input_size`, `color_order`, `pixel_scale`,
`mean`, `std`, and `alignment` must be set for the selected model. `mean` and
`std` are three-element values in the same pixel domain used after multiplying
by `pixel_scale`; the service applies `(pixel * pixel_scale - mean) / std`.
The application selects a model input/output tensor automatically only when
there is exactly one of each. Otherwise set the tensor names explicitly.

`alignment: five_point` requires five configured reference points in the schema
order (left eye, right eye, nose tip, left mouth corner, right mouth corner).
`alignment: none` resizes square images to `input_size`. When
`honor_preprocessed_flag` is true (the default), an image marked
`is_preprocessed` skips resizing/alignment but still receives configured color
conversion and pixel normalization. Such an image must already match
`input_size`. Set `honor_preprocessed_flag: false` to always run the configured
spatial preprocessing.

BMP (`encoding=3`), raw RGB (`encoding=1`), and PNG (`encoding=5`) payloads
are accepted. Images must be square. A model's output
dimension is discovered at startup, so embeddings are not fixed to 192, 512,
or any other hard-coded size.

## Collections and matching

* `raw_enrol` stores a normalized embedding per enrollment, along with
  `user_id`, `name`, `custom`, `time_unix_ms`, and the original image bytes as
  a base64 string in the `image` payload field.
* `normalized_enrol` stores one L2-normalized arithmetic mean per user. Each
  enrollment removes the previous mean and recalculates it from the user's
  L2-normalized raw vectors.
* A changed enrollment name is applied to all of that user's raw points and
  their normalized point.
* User deletion removes matching points from both collections. User-info
  returns every raw embedding and timestamp.

Set `qdrant.metric` to `cosine` or `euclidean`. Existing collections are
validated against the model's output dimension and configured metric; a
mismatch is reported and the service will not recreate or erase the collection.
The Qdrant client prefers gRPC for operations and falls back to REST when the
configured gRPC endpoint is unavailable. Startup compatibility checks are
disabled to avoid extra endpoint probes.

Query candidates include `raw_distance` and `confidence`. Confidence uses the
following bounded similarity mappings for unit-length vectors:

* Cosine: `1 - cosine_distance / 2`
* Euclidean: `1 - euclidean_distance ** 2 / 4`

The value is clamped to 0..1 and is a similarity score, not a statistically
calibrated probability. The query's `min_confidence` filters on this value.
The schema's `confidence` comment and the additional `raw_distance` field are
defined in [schemas/faceid.fbs](./schemas/faceid.fbs); regenerate client bindings
from that schema when updating consumers.

## NATS subjects and messages

By default the service subscribes to:

* `faceid.enrol`
* `faceid.query`
* `faceid.user_info`
* `faceid.delete`
* `faceid.debug`

Subjects are configurable. Enrolment, query, user-info, delete, and debug
responses are published to the request's `reply_to` when supplied. The
FlatBuffer union body must match the operation's subject. Malformed messages
and operation errors are logged; valid requests with a reply subject receive
their schema's failure response.

NATS supports an optional token, username/password, or credentials file.
Qdrant supports an optional API key. Keep real credentials out of version
control.

## Tests

With the test extra installed, run:

```sh
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```






# Run Qdrant
```sh
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```