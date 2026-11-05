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

std::vector<int64_t> tensorShape(const ModelConfig& config) {
  if (config.input_layout == "NHWC") {
    return {1, config.input_size, config.input_size, 3};
  }
  return {1, 3, config.input_size, config.input_size};
}

double pointCoordinate(double value) {
  return std::isfinite(value) ? value :
      throw InferenceError("The supplied keypoints must be finite");
}

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

InferenceError::InferenceError(const std::string& message)
    : std::runtime_error(message) {}

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

  std::vector<float> infer(const std::vector<float>& tensor) {
    if (config.runtime == "onnx") {
      return inferOnnx(tensor);
    }
    return inferLiteRt(tensor);
  }

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

EmbeddingModel::EmbeddingModel(const ModelConfig& config)
    : impl_(new Impl(config)) {}

EmbeddingModel::~EmbeddingModel() {}

std::vector<float> EmbeddingModel::embed(const ImageRequest& image,
                                         InferenceTiming* timing) {
  return impl_->run(image, timing);
}

size_t EmbeddingModel::embeddingDimension() const { return impl_->dimension; }

const std::string& EmbeddingModel::runtimeName() const {
  return impl_->config.runtime;
}

const std::string& EmbeddingModel::modelPath() const { return impl_->config.path; }

}  // namespace faceid