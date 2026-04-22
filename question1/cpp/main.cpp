#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

namespace {

struct Args {
  std::string onnx_dir = "../artifacts/onnx";
  std::string prompt_ids = "../artifacts/prompt_ids.json";
  std::string report_path = "../artifacts/reports/cpp_inference_report.txt";
  std::string output_image = "../artifacts/images/cpp_onnx_output.ppm";
  int height = 512;
  int width = 512;
  int num_steps = 20;
  float guidance_scale = 7.5f;
  int seed = 42;
  int runs = 5;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  std::unordered_map<std::string, std::string*> string_flags = {
      {"--onnx-dir", &args.onnx_dir},
      {"--prompt-ids", &args.prompt_ids},
      {"--report", &args.report_path},
      {"--output-image", &args.output_image},
  };

  for (int index = 1; index < argc; ++index) {
    std::string key = argv[index];
    if (string_flags.count(key) > 0 && index + 1 < argc) {
      *string_flags[key] = argv[++index];
      continue;
    }

    if (index + 1 >= argc) {
      throw std::runtime_error("Missing value for argument: " + key);
    }

    std::string value = argv[++index];
    if (key == "--height") {
      args.height = std::stoi(value);
    } else if (key == "--width") {
      args.width = std::stoi(value);
    } else if (key == "--steps") {
      args.num_steps = std::stoi(value);
    } else if (key == "--guidance-scale") {
      args.guidance_scale = std::stof(value);
    } else if (key == "--seed") {
      args.seed = std::stoi(value);
    } else if (key == "--runs") {
      args.runs = std::stoi(value);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  return args;
}

std::vector<double> MakeBetas(int num_train_timesteps, double beta_start, double beta_end) {
  std::vector<double> betas(num_train_timesteps);
  double start = std::sqrt(beta_start);
  double end = std::sqrt(beta_end);

  for (int index = 0; index < num_train_timesteps; ++index) {
    double ratio = static_cast<double>(index) / static_cast<double>(num_train_timesteps - 1);
    double value = start + ratio * (end - start);
    betas[index] = value * value;
  }

  return betas;
}

std::vector<double> MakeAlphasCumprod(const std::vector<double>& betas) {
  std::vector<double> alphas_cumprod(betas.size());
  double running = 1.0;
  for (std::size_t index = 0; index < betas.size(); ++index) {
    running *= (1.0 - betas[index]);
    alphas_cumprod[index] = running;
  }
  return alphas_cumprod;
}

std::vector<int64_t> MakeTimesteps(int num_train_timesteps, int num_inference_steps) {
  std::vector<int64_t> timesteps(num_inference_steps);
  for (int index = 0; index < num_inference_steps; ++index) {
    double ratio = static_cast<double>(index) / static_cast<double>(num_inference_steps - 1);
    double raw = ratio * static_cast<double>(num_train_timesteps - 1);
    timesteps[num_inference_steps - 1 - index] = static_cast<int64_t>(std::llround(raw));
  }
  return timesteps;
}

std::vector<float> DdimStep(
    const std::vector<float>& noise_pred,
    int timestep_index,
    const std::vector<float>& sample,
    const std::vector<double>& alphas_cumprod,
    const std::vector<int64_t>& timesteps) {
  int64_t timestep = timesteps[timestep_index];
  int64_t prev_timestep = timestep_index + 1 < static_cast<int>(timesteps.size()) ? timesteps[timestep_index + 1] : -1;

  double alpha_prod_t = alphas_cumprod[static_cast<std::size_t>(timestep)];
  double alpha_prod_prev = prev_timestep < 0 ? 1.0 : alphas_cumprod[static_cast<std::size_t>(prev_timestep)];
  double beta_prod_t = 1.0 - alpha_prod_t;

  std::vector<float> previous(sample.size());
  for (std::size_t index = 0; index < sample.size(); ++index) {
    double pred_original = (static_cast<double>(sample[index]) - std::sqrt(beta_prod_t) * static_cast<double>(noise_pred[index])) /
                           std::sqrt(alpha_prod_t);
    double pred_direction = std::sqrt(1.0 - alpha_prod_prev) * static_cast<double>(noise_pred[index]);
    previous[index] = static_cast<float>(std::sqrt(alpha_prod_prev) * pred_original + pred_direction);
  }
  return previous;
}

std::vector<int64_t> ReadIds(const nlohmann::json& payload, const std::string& key) {
  return payload.at(key).get<std::vector<int64_t>>();
}

std::vector<float> RunTextEncoder(
    Ort::Session& session,
    const std::vector<int64_t>& input_ids,
    std::vector<int64_t>& output_shape) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto output_name = session.GetOutputNameAllocated(0, allocator);

  std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info,
      const_cast<int64_t*>(input_ids.data()),
      input_ids.size(),
      input_shape.data(),
      input_shape.size());

  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

  auto& output_tensor = output_tensors.front();
  auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
  output_shape = tensor_info.GetShape();
  float* raw = output_tensor.GetTensorMutableData<float>();
  std::size_t total = tensor_info.GetElementCount();
  return std::vector<float>(raw, raw + total);
}

std::vector<float> RunUnet(
    Ort::Session& session,
    const std::vector<float>& latents,
    const std::vector<int64_t>& latent_shape,
    int64_t timestep,
    const std::vector<float>& encoder_hidden_states,
    const std::vector<int64_t>& encoder_shape) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto latents_name = session.GetInputNameAllocated(0, allocator);
  auto timestep_name = session.GetInputNameAllocated(1, allocator);
  auto hidden_name = session.GetInputNameAllocated(2, allocator);
  auto output_name = session.GetOutputNameAllocated(0, allocator);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value latents_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float*>(latents.data()),
      latents.size(),
      latent_shape.data(),
      latent_shape.size());

  std::vector<int64_t> timestep_shape = {1};
  Ort::Value timestep_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info,
      &timestep,
      1,
      timestep_shape.data(),
      timestep_shape.size());

  Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float*>(encoder_hidden_states.data()),
      encoder_hidden_states.size(),
      encoder_shape.data(),
      encoder_shape.size());

  std::array<const char*, 3> input_names = {latents_name.get(), timestep_name.get(), hidden_name.get()};
  std::array<Ort::Value, 3> inputs = {std::move(latents_tensor), std::move(timestep_tensor), std::move(hidden_tensor)};
  const char* output_names[] = {output_name.get()};

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), output_names, 1);
  auto& output_tensor = outputs.front();
  auto info = output_tensor.GetTensorTypeAndShapeInfo();
  float* raw = output_tensor.GetTensorMutableData<float>();
  std::size_t total = info.GetElementCount();
  return std::vector<float>(raw, raw + total);
}

std::vector<float> RunVaeDecoder(Ort::Session& session, std::vector<float>& latents, const std::vector<int64_t>& latent_shape) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto output_name = session.GetOutputNameAllocated(0, allocator);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      latents.data(),
      latents.size(),
      latent_shape.data(),
      latent_shape.size());

  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
  auto& output_tensor = outputs.front();
  auto info = output_tensor.GetTensorTypeAndShapeInfo();
  float* raw = output_tensor.GetTensorMutableData<float>();
  std::size_t total = info.GetElementCount();
  return std::vector<float>(raw, raw + total);
}

std::vector<float> ConcatenateLatents(const std::vector<float>& latents) {
  std::vector<float> duplicated;
  duplicated.reserve(latents.size() * 2);
  duplicated.insert(duplicated.end(), latents.begin(), latents.end());
  duplicated.insert(duplicated.end(), latents.begin(), latents.end());
  return duplicated;
}

std::vector<float> ConcatenateHiddenStates(const std::vector<float>& first, const std::vector<float>& second) {
  std::vector<float> combined;
  combined.reserve(first.size() + second.size());
  combined.insert(combined.end(), first.begin(), first.end());
  combined.insert(combined.end(), second.begin(), second.end());
  return combined;
}

std::vector<float> ApplyGuidance(const std::vector<float>& noise_pred, float guidance_scale) {
  std::size_t half = noise_pred.size() / 2;
  std::vector<float> guided(half);
  for (std::size_t index = 0; index < half; ++index) {
    float uncond = noise_pred[index];
    float text = noise_pred[index + half];
    guided[index] = uncond + guidance_scale * (text - uncond);
  }
  return guided;
}

void SavePpm(const std::vector<float>& decoded, const std::string& path, int height, int width) {
  std::ofstream handle(path, std::ios::binary);
  handle << "P6\n" << width << " " << height << "\n255\n";

  int image_plane = height * width;
  for (int pixel = 0; pixel < image_plane; ++pixel) {
    for (int channel = 0; channel < 3; ++channel) {
      float value = decoded[static_cast<std::size_t>(channel * image_plane + pixel)];
      value = std::clamp((value / 2.0f) + 0.5f, 0.0f, 1.0f);
      unsigned char byte = static_cast<unsigned char>(std::round(value * 255.0f));
      handle.write(reinterpret_cast<const char*>(&byte), 1);
    }
  }
}

void WriteReport(const Args& args, double average_latency_seconds, const std::string& prompt) {
  std::ofstream handle(args.report_path);
  handle << "prompt=" << prompt << "\n";
  handle << "num_steps=" << args.num_steps << "\n";
  handle << "runs=" << args.runs << "\n";
  handle << "seed_start=" << args.seed << "\n";
  handle << "average_latency_seconds=" << average_latency_seconds << "\n";
  handle << "output_image=" << args.output_image << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);

    std::ifstream prompt_file(args.prompt_ids);
    nlohmann::json prompt_payload = nlohmann::json::parse(prompt_file);
    std::vector<int64_t> prompt_ids = ReadIds(prompt_payload, "prompt_ids");
    std::vector<int64_t> negative_prompt_ids = ReadIds(prompt_payload, "negative_prompt_ids");

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "question1_onnx_cpp");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session text_encoder(env, (args.onnx_dir + "/text_encoder.onnx").c_str(), session_options);
    Ort::Session unet(env, (args.onnx_dir + "/unet.onnx").c_str(), session_options);
    Ort::Session vae_decoder(env, (args.onnx_dir + "/vae_decoder.onnx").c_str(), session_options);

    std::vector<int64_t> text_shape;
    std::vector<float> unconditional_hidden = RunTextEncoder(text_encoder, negative_prompt_ids, text_shape);
    std::vector<float> conditional_hidden = RunTextEncoder(text_encoder, prompt_ids, text_shape);
    std::vector<float> encoder_hidden_states = ConcatenateHiddenStates(unconditional_hidden, conditional_hidden);
    std::vector<int64_t> encoder_shape = {2, text_shape[1], text_shape[2]};

    std::vector<double> betas = MakeBetas(1000, 0.00085, 0.012);
    std::vector<double> alphas_cumprod = MakeAlphasCumprod(betas);
    std::vector<int64_t> timesteps = MakeTimesteps(1000, args.num_steps);

    std::vector<double> latencies;
    int latent_height = args.height / 8;
    int latent_width = args.width / 8;
    int latent_size = 4 * latent_height * latent_width;
    std::vector<int64_t> latent_shape = {2, 4, latent_height, latent_width};
    std::vector<int64_t> decoder_shape = {1, 4, latent_height, latent_width};

    for (int run_index = 0; run_index < args.runs; ++run_index) {
      std::mt19937 generator(args.seed + run_index);
      std::normal_distribution<float> normal(0.0f, 1.0f);
      std::vector<float> latents(static_cast<std::size_t>(latent_size));
      for (float& value : latents) {
        value = normal(generator);
      }

      auto start = std::chrono::steady_clock::now();
      for (int timestep_index = 0; timestep_index < args.num_steps; ++timestep_index) {
        std::vector<float> duplicated_latents = ConcatenateLatents(latents);
        std::vector<float> noise_pred = RunUnet(
            unet,
            duplicated_latents,
            latent_shape,
            timesteps[timestep_index],
            encoder_hidden_states,
            encoder_shape);
        std::vector<float> guided_noise = ApplyGuidance(noise_pred, args.guidance_scale);
        latents = DdimStep(guided_noise, timestep_index, latents, alphas_cumprod, timesteps);
      }

      for (float& value : latents) {
        value /= 0.18215f;
      }
      std::vector<float> decoded = RunVaeDecoder(vae_decoder, latents, decoder_shape);
      auto end = std::chrono::steady_clock::now();

      double latency = std::chrono::duration<double>(end - start).count();
      latencies.push_back(latency);

      if (run_index == args.runs - 1) {
        SavePpm(decoded, args.output_image, args.height, args.width);
      }
    }

    double average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(latencies.size());
    WriteReport(args, average_latency, prompt_payload.at("prompt").get<std::string>());

    std::cout << "c++ average latency: " << average_latency << " seconds\n";
    std::cout << "c++ output image saved to: " << args.output_image << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
