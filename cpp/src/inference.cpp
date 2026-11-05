#include "faceid/inference.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace faceid {
namespace {

/**
 * Build the batch-one tensor dimensions expected by the configured model.
 * @param config Model input layout and square image size.
 * @return Shape in NHWC or NCHW order.
 */
std::vector<int64_t> tensorShape(const ModelConfig& config) {
  if (config.input_layout == "NHWC") {
    return {1, config.input_size, config.input_size, 3};
  }
  return {1, 3, config.input_size, config.input_size};
}

/** Validate a landmark coordinate before it enters the alignment transform. */
double pointCoordinate(double value) {
  return std::isfinite(value) ? value :
      throw InferenceError("The supplied keypoints must be finite");
}

/**
 * Warp an RGB face crop to the model's five configured landmark positions.
 * @param image Decoded source image.
 * @param source_points Request landmarks in left-eye-to-right-mouth order.
 * @param target_points Corresponding model-space reference landmarks.
 * @param size Output crop width and height in pixels.
 * @return Affine-warped square image; invalid or degenerate landmarks throw.
 */
cv::Mat alignFivePoints(const cv::Mat& image,
                        const std::vector<std::array<float, 2> >& source_points,
                        const std::vector<std::array<float, 2> >& target_points,
                        int size) {
  if (source_points.size() != 5 || target_points.size() != 5) {
    throw InferenceError(
        "Five-point alignment requires exactly five request keypoints and "
        "configured reference_keypoints");
  }

  double source_x = 0.0;
  double source_y = 0.0;
  double target_x = 0.0;
  double target_y = 0.0;
  for (size_t index = 0; index < 5; ++index) {
    source_x += pointCoordinate(source_points[index][0]);
    source_y += pointCoordinate(source_points[index][1]);
    target_x += pointCoordinate(target_points[index][0]);
    target_y += pointCoordinate(target_points[index][1]);
  }
  source_x /= 5.0;
  source_y /= 5.0;
  target_x /= 5.0;
  target_y /= 5.0;

  double denominator = 0.0;
  double cosine_scale = 0.0;
  double sine_scale = 0.0;
  for (size_t index = 0; index < 5; ++index) {
    const double x = source_points[index][0] - source_x;
    const double y = source_points[index][1] - source_y;
    const double u = target_points[index][0] - target_x;
    const double v = target_points[index][1] - target_y;
    denominator += x * x + y * y;
    cosine_scale += x * u + y * v;
    sine_scale += x * v - y * u;
  }
  if (denominator == 0.0) {
    throw InferenceError(
        "The supplied keypoints do not define a usable alignment");
  }
  cosine_scale /= denominator;
  sine_scale /= denominator;
  if (!std::isfinite(cosine_scale) || !std::isfinite(sine_scale) ||
      cosine_scale * cosine_scale + sine_scale * sine_scale <= 0.0) {
    throw InferenceError("The supplied keypoints produce an invalid alignment");
  }

  const cv::Matx23d transform(
      cosine_scale, -sine_scale,
      target_x - cosine_scale * source_x + sine_scale * source_y,
      sine_scale, cosine_scale,
      target_y - sine_scale * source_x - cosine_scale * source_y);
  cv::Mat aligned;
  cv::warpAffine(image, aligned, transform, cv::Size(size, size),
                 cv::INTER_CUBIC, cv::BORDER_CONSTANT);
  return aligned;
}

/**
 * Validate and decode supported compressed image bytes as RGB.
 * @param image Encoded request data and its declared image format.
 * @param decoded Output matrix populated with decoded RGB pixels.
 * @throws InferenceError for format mismatch, unsupported encodings, or decode
 *         failure.
 */
void decodeCompressedImage(const ImageRequest& image, cv::Mat* decoded) {
  const std::vector<uint8_t>& bytes = image.bytes;
  const bool bmp = bytes.size() >= 2 && bytes[0] == 'B' && bytes[1] == 'M';
  const bool png = bytes.size() >= 8 && bytes[0] == 0x89 && bytes[1] == 'P' &&
                   bytes[2] == 'N' && bytes[3] == 'G' && bytes[4] == 0x0d &&
                   bytes[5] == 0x0a && bytes[6] == 0x1a && bytes[7] == 0x0a;
  if (image.encoding == faceid::fbs::ImageEncoding::Bmp ||
      image.encoding == faceid::fbs::ImageEncoding::Unknown) {
    if (!bmp) {
      throw InferenceError("The image payload is not BMP data");
    }
  } else if (image.encoding == faceid::fbs::ImageEncoding::Png) {
    if (!png) {
      throw InferenceError("The image payload is not PNG data");
    }
  } else {
    throw InferenceError("Unsupported image encoding");
  }

  cv::Mat encoded(1, static_cast<int>(bytes.size()), CV_8UC1,
                  const_cast<uint8_t*>(bytes.data()));
  *decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (decoded->empty()) {
    throw InferenceError("Unable to decode image data");
  }
  cv::cvtColor(*decoded, *decoded, cv::COLOR_BGR2RGB);
}

/**
 * Resolve a model input/output tensor by configured name or sole-candidate rule.
 * @param names Tensor names reported by the selected inference runtime.
 * @param configured_name Optional explicit name from application configuration.
 * @param kind "input" or "output", used in diagnostics and setting names.
 * @param selected_name Receives the resolved tensor name.
 * @param selected_index Receives that tensor's position in the runtime list.
 */
void selectTensorNames(const std::vector<std::string>& names,
                       const std::string& configured_name, const char* kind,
                       std::string* selected_name, size_t* selected_index) {
  if (!configured_name.empty()) {
    for (size_t index = 0; index < names.size(); ++index) {
      if (names[index] == configured_name) {
        *selected_name = configured_name;
        *selected_index = index;
        return;
      }
    }
    throw InferenceError(std::string("Configured model ") + kind +
                         " tensor '" + configured_name + "' was not found");
  }
  if (names.size() != 1) {
    std::ostringstream message;
    message << "Model has " << names.size() << " " << kind
            << " tensors; configure model." << kind << "_name";
    throw InferenceError(message.str());
  }
  *selected_name = names[0];
  *selected_index = 0;
}

/**
 * Flatten a LiteRT delegate-options YAML mapping into alternating key/value text.
 * @param yaml_options Serialized YAML mapping retained by configuration loading.
 * @return Strings consumed by the external delegate API in key/value pairs.
 */
std::vector<std::string> optionStrings(const std::string& yaml_options) {
  YAML::Node options;
  try {
    options = YAML::Load(yaml_options);
  } catch (const YAML::Exception& error) {
    throw InferenceError(std::string("Invalid LiteRT delegate options: ") +
                         error.what());
  }
  if (!options.IsMap()) {
    throw InferenceError("LiteRT delegate options must be a mapping");
  }
  std::vector<std::string> values;
  for (YAML::const_iterator item = options.begin(); item != options.end(); ++item) {
    values.push_back(item->first.as<std::string>());
    values.push_back(YAML::Dump(item->second));
  }
  return values;
}

}  // namespace

/** Construct the inference exception used for invalid inputs and runtime errors. */
InferenceError::InferenceError(const std::string& message)
    : std::runtime_error(message) {}

/**
 * Validate and scale a model output to unit Euclidean length.
 * @param values Raw model output values.
 * @return Finite, nonzero values normalized to unit length.
 * @throws InferenceError for empty, non-finite, or zero-norm output.
 */
std::vector<float> normalizeEmbedding(const std::vector<float>& values) {
  if (values.empty()) {
    throw InferenceError("The model returned an empty embedding");
  }
  double squared_norm = 0.0;
  for (size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      throw InferenceError("The model returned a non-finite embedding");
    }
    squared_norm += static_cast<double>(values[index]) * values[index];
  }
  const double norm = std::sqrt(squared_norm);
  if (!std::isfinite(norm) || norm == 0.0) {
    throw InferenceError("The model returned a zero-length embedding");
  }
  std::vector<float> normalized(values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    normalized[index] = static_cast<float>(values[index] / norm);
  }
  return normalized;
}

/**
 * Convert request image bytes into the normalized model input tensor.
 *
 * Raw RGB data is copied directly; supported BMP/PNG payloads are decoded and
 * converted from OpenCV's BGR ordering. The image is checked for square
 * geometry, optionally aligned or resized, optionally reordered to BGR, then
 * normalized and laid out as NCHW or NHWC.
 * @param request Image bytes, dimensions, encoding, preprocessing flag, and
 *                optional five-point landmarks.
 * @param config Model size, channel order, layout, normalization, and alignment.
 * @return Contiguous float tensor ready for ONNX Runtime or LiteRT.
 */
std::vector<float> prepareImage(const ImageRequest& request,
                                const ModelConfig& config) {
  cv::Mat rgb;
  if (request.encoding == faceid::fbs::ImageEncoding::Rgb) {
    if (request.width == 0 || request.height == 0) {
      throw InferenceError(
          "RGB image data requires positive message width and height");
    }
    const uint64_t expected_length =
        static_cast<uint64_t>(request.width) * request.height * 3;
    if (request.bytes.size() != expected_length) {
      throw InferenceError(
          "RGB image data length does not match message width and height");
    }
    cv::Mat view(static_cast<int>(request.height), static_cast<int>(request.width),
                 CV_8UC3, const_cast<uint8_t*>(request.bytes.data()));
    rgb = view.clone();
  } else {
    decodeCompressedImage(request, &rgb);
  }

  if (rgb.cols != rgb.rows) {
    throw InferenceError("Face images must be square");
  }
  if (request.width != 0 && request.width != static_cast<uint32_t>(rgb.cols)) {
    throw InferenceError("Message width does not match decoded image width");
  }
  if (request.height != 0 && request.height != static_cast<uint32_t>(rgb.rows)) {
    throw InferenceError("Message height does not match decoded image height");
  }

  const bool skip_spatial =
      request.is_preprocessed && config.honor_preprocessed_flag;
  if (skip_spatial) {
    if (rgb.cols != config.input_size || rgb.rows != config.input_size) {
      throw InferenceError(
          "An image marked preprocessed must match model.input_size when "
          "spatial preprocessing is skipped");
    }
  } else if (config.alignment == "five_point") {
    rgb = alignFivePoints(rgb, request.keypoints, config.reference_keypoints,
                          config.input_size);
  } else {
    cv::resize(rgb, rgb, cv::Size(config.input_size, config.input_size), 0.0, 0.0,
               cv::INTER_LINEAR);
  }

  if (config.color_order == "BGR") {
    cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
  }
  const int channels = 3;
  const int size = config.input_size;
  std::vector<float> tensor(static_cast<size_t>(channels) * size * size);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
      for (int channel = 0; channel < channels; ++channel) {
        const size_t index = config.input_layout == "NCHW"
                                 ? (static_cast<size_t>(channel) * size + y) * size + x
                                 : (static_cast<size_t>(y) * size + x) * channels + channel;
        tensor[index] =
            (static_cast<float>(pixel[channel]) * config.pixel_scale -
             config.mean[channel]) /
            config.stddev[channel];
      }
    }
  }
  return tensor;
}

class EmbeddingModel::Impl {
 public:
  /**
   * Load one configured inference backend and discover the embedding dimension.
   * @param model_config Validated model path, runtime, and input metadata.
   * @throws InferenceError if model loading, tensor selection, or warm-up fails.
   */
  explicit Impl(const ModelConfig& model_config)
      : config(model_config), dimension(0), input_index(0), output_index(0) {
    std::ifstream model_file(config.path.c_str(), std::ios::binary);
    if (!model_file.good()) {
      throw InferenceError("Model file does not exist: " + config.path);
    }
    if (config.runtime == "onnx") {
      loadOnnx();
    } else {
      loadLiteRt();
    }
    const std::vector<float> output = infer(std::vector<float>(
        static_cast<size_t>(3) * config.input_size * config.input_size, 0.0f));
    dimension = output.size();
    if (dimension == 0) {
      throw InferenceError("Model output has no embedding values");
    }
  }

  /**
   * Run the full image-to-embedding flow for one request.
   * @param image Request pixels and their preprocessing metadata.
   * @param timing Optional output for preprocessing and inference durations;
   *               lock wait is excluded.
   * @return Unit-length embedding with the dimension discovered at startup.
   */
  std::vector<float> run(const ImageRequest& image, InferenceTiming* timing) {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    const Clock::time_point preprocess_start = Clock::now();
    const std::vector<float> tensor = prepareImage(image, config);
    const Clock::time_point preprocess_end = Clock::now();
    std::lock_guard<std::mutex> lock(mutex);
    const Clock::time_point inference_start = Clock::now();
    std::vector<float> output = normalizeEmbedding(infer(tensor));
    const Clock::time_point inference_end = Clock::now();
    if (timing != nullptr) {
      timing->preprocess_ms =
          Milliseconds(preprocess_end - preprocess_start).count();
      timing->inference_ms =
          Milliseconds(inference_end - inference_start).count();
    }
    if (output.size() != dimension) {
      std::ostringstream message;
      message << "Model output dimension changed from " << dimension << " to "
              << output.size();
      throw InferenceError(message.str());
    }
    return output;
  }

  /**
   * Dispatch an already prepared float tensor to the configured backend.
   * @param tensor Contiguous normalized tensor in configured NCHW/NHWC layout.
   * @return Raw float output from the selected ONNX or LiteRT tensor.
   */
  std::vector<float> infer(const std::vector<float>& tensor) {
    if (config.runtime == "onnx") {
      return inferOnnx(tensor);
    }
    return inferLiteRt(tensor);
  }

  /**
   * Create the ONNX Runtime environment/session and resolve input/output names.
   * @throws InferenceError when ONNX loading or tensor selection fails.
   */
  void loadOnnx() {
    try {
      ort_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "imx-faceid"));
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      ort_session.reset(new Ort::Session(*ort_env, config.path.c_str(), options));
      Ort::AllocatorWithDefaultOptions allocator;
      std::vector<std::string> input_names;
      std::vector<std::string> output_names;
      for (size_t index = 0; index < ort_session->GetInputCount(); ++index) {
        Ort::AllocatedStringPtr name =
            ort_session->GetInputNameAllocated(index, allocator);
        input_names.push_back(name.get());
      }
      for (size_t index = 0; index < ort_session->GetOutputCount(); ++index) {
        Ort::AllocatedStringPtr name =
            ort_session->GetOutputNameAllocated(index, allocator);
        output_names.push_back(name.get());
      }
      selectTensorNames(input_names, config.input_name, "input", &input_name,
                        &input_index);
      selectTensorNames(output_names, config.output_name, "output", &output_name,
                        &output_index);
    } catch (const Ort::Exception& error) {
      throw InferenceError(std::string("Unable to initialize ONNX model: ") +
                           error.what());
    }
  }

  /**
   * Build and allocate a LiteRT interpreter, optionally installing a delegate.
   * @throws InferenceError when model loading, delegate setup, or tensor
   *         allocation fails.
   */
  void loadLiteRt() {
    model = tflite::FlatBufferModel::BuildFromFile(config.path.c_str());
    if (!model) {
      throw InferenceError("Unable to load LiteRT model: " + config.path);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder.SetNumThreads(1);
    if (!config.delegate_library.empty()) {
      delegate_options = TfLiteExternalDelegateOptionsDefault(
          config.delegate_library.c_str());
      delegate_option_strings = optionStrings(config.delegate_options);
      for (size_t index = 0; index + 1 < delegate_option_strings.size(); index += 2) {
        if (delegate_options.insert(&delegate_options,
                                    delegate_option_strings[index].c_str(),
                                    delegate_option_strings[index + 1].c_str()) !=
            kTfLiteOk) {
          throw InferenceError("Unable to set LiteRT delegate option");
        }
      }
      delegate.reset(TfLiteExternalDelegateCreate(&delegate_options));
      if (!delegate) {
        throw InferenceError("Unable to initialize LiteRT delegate: " +
                             config.delegate_library);
      }
      builder.AddDelegate(delegate.get());
    }
    if (builder(&interpreter) != kTfLiteOk || !interpreter) {
      throw InferenceError("Unable to initialize LiteRT interpreter");
    }

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (size_t index = 0; index < interpreter->inputs().size(); ++index) {
      const TfLiteTensor* tensor = interpreter->tensor(interpreter->inputs()[index]);
      input_names.push_back(tensor->name == nullptr ? "" : tensor->name);
    }
    for (size_t index = 0; index < interpreter->outputs().size(); ++index) {
      const TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[index]);
      output_names.push_back(tensor->name == nullptr ? "" : tensor->name);
    }
    selectTensorNames(input_names, config.input_name, "input", &input_name,
                      &input_index);
    selectTensorNames(output_names, config.output_name, "output", &output_name,
                      &output_index);
    const std::vector<int> input_shape = {
        1, config.input_layout == "NCHW" ? 3 : config.input_size,
        config.input_layout == "NCHW" ? config.input_size : config.input_size,
        config.input_layout == "NCHW" ? config.input_size : 3};
    if (interpreter->ResizeInputTensor(interpreter->inputs()[input_index],
                                       input_shape) != kTfLiteOk ||
        interpreter->AllocateTensors() != kTfLiteOk) {
      throw InferenceError("Unable to allocate LiteRT model tensors");
    }
  }

  /**
   * Wrap an input tensor, execute the selected ONNX graph output, and copy it.
   * @param tensor Normalized input values produced by prepareImage.
   * @return Output tensor elements as owned float values.
   */
  std::vector<float> inferOnnx(const std::vector<float>& tensor) {
    const std::vector<int64_t> shape = tensorShape(config);
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(tensor.data()), tensor.size(),
        shape.data(), shape.size());
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};
    try {
      std::vector<Ort::Value> outputs = ort_session->Run(
          Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
      const Ort::TensorTypeAndShapeInfo info =
          outputs[0].GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw InferenceError("ONNX model output must use float32 values");
      }
      const size_t count = info.GetElementCount();
      const float* values = outputs[0].GetTensorData<float>();
      return std::vector<float>(values, values + count);
    } catch (const Ort::Exception& error) {
      throw InferenceError(std::string("ONNX inference failed: ") + error.what());
    }
  }

  /**
   * Populate LiteRT input storage, invoke the interpreter, and decode its output.
   * Quantized tensors are converted using their per-tensor scale and zero point.
   * @param tensor Normalized input values produced by prepareImage.
   * @return Output tensor converted to float values for embedding normalization.
   */
  std::vector<float> inferLiteRt(const std::vector<float>& tensor) {
    TfLiteTensor* input =
        interpreter->tensor(interpreter->inputs()[input_index]);
    const size_t count = tensor.size();
    if (input->type == kTfLiteFloat32) {
      std::memcpy(input->data.f, tensor.data(), count * sizeof(float));
    } else if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
      const float scale = input->params.scale;
      if (!std::isfinite(scale) || scale <= 0.0f) {
        throw InferenceError(
            "LiteRT quantized input must use per-tensor quantization with a "
            "positive scale");
      }
      for (size_t index = 0; index < count; ++index) {
        const double quantized =
            std::nearbyint(tensor[index] / scale + input->params.zero_point);
        if (input->type == kTfLiteUInt8) {
          input->data.uint8[index] = static_cast<uint8_t>(std::max(
              0.0, std::min(255.0, quantized)));
        } else {
          input->data.int8[index] = static_cast<int8_t>(std::max(
              -128.0, std::min(127.0, quantized)));
        }
      }
    } else {
      throw InferenceError("Unsupported LiteRT input tensor type");
    }
    if (interpreter->Invoke() != kTfLiteOk) {
      throw InferenceError("LiteRT inference failed");
    }

    const TfLiteTensor* output =
        interpreter->tensor(interpreter->outputs()[output_index]);
    const size_t output_count = output->bytes /
                                (output->type == kTfLiteFloat32 ? sizeof(float) : 1);
    std::vector<float> result(output_count);
    if (output->type == kTfLiteFloat32) {
      std::copy(output->data.f, output->data.f + output_count, result.begin());
    } else if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
      const float scale = output->params.scale;
      if (!std::isfinite(scale) || scale <= 0.0f) {
        throw InferenceError(
            "LiteRT quantized output must use per-tensor quantization with a "
            "positive scale");
      }
      for (size_t index = 0; index < output_count; ++index) {
        const int value = output->type == kTfLiteUInt8
                              ? output->data.uint8[index]
                              : output->data.int8[index];
        result[index] =
            (value - output->params.zero_point) * scale;
      }
    } else {
      throw InferenceError("Unsupported LiteRT output tensor type");
    }
    return result;
  }

  ModelConfig config;
  size_t dimension;
  size_t input_index;
  size_t output_index;
  std::string input_name;
  std::string output_name;
  std::mutex mutex;

  std::unique_ptr<Ort::Env> ort_env;
  std::unique_ptr<Ort::Session> ort_session;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate{
      nullptr, TfLiteExternalDelegateDelete};
  TfLiteExternalDelegateOptions delegate_options{};
  std::vector<std::string> delegate_option_strings;
  std::unique_ptr<tflite::Interpreter> interpreter;
};

/** Construct the owning implementation and initialize the selected backend. */
EmbeddingModel::EmbeddingModel(const ModelConfig& config)
    : impl_(new Impl(config)) {}

/** Release the backend through the implementation object's RAII-owned members. */
EmbeddingModel::~EmbeddingModel() {}

/**
 * Generate a normalized embedding and optionally report stage timings.
 * @param image Encoded or raw request pixels and associated metadata.
 * @param timing Optional destination for preprocessing and inference durations.
 * @return Unit-length model embedding.
 */
std::vector<float> EmbeddingModel::embed(const ImageRequest& image,
                                         InferenceTiming* timing) {
  return impl_->run(image, timing);
}

/** Return the output vector length measured during model warm-up. */
size_t EmbeddingModel::embeddingDimension() const { return impl_->dimension; }

/** Return the configured backend identifier ("onnx" or "litert"). */
const std::string& EmbeddingModel::runtimeName() const {
  return impl_->config.runtime;
}

/** Return the resolved model file path retained at initialization. */
const std::string& EmbeddingModel::modelPath() const { return impl_->config.path; }

}  // namespace faceid