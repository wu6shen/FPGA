/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#include <ctime>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdio>

#define NO_STRICT
#define CNN_USE_CAFFE_CONVERTER

#ifndef DNN_USE_IMAGE_API
#define DNN_USE_IMAGE_API
#endif

#include "tiny_dnn/tiny_dnn.h"

tiny_dnn::image<float> compute_mean(int width,
                                    int height) {
	/** caffe convert
  caffe::BlobProto blob;
  tiny_dnn::detail::read_proto_from_binary(mean_file, &blob);

  auto data = blob.mutable_data()->mutable_data();
  */

  float *data = new float[width * height * 3];
  std::ifstream file;
  file.open("./model/image_mean");
  for (int i = 0; i < width * height * 3; i++) {
	  file >> *(data + i);
  }
  file.close();

  tiny_dnn::image<float> original(data, width, height, 
                                  tiny_dnn::image_type::bgr);

  return mean_image(original);
}

void preprocess(const tiny_dnn::image<float> &img,
                const tiny_dnn::image<float> &mean,
                int width,
                int height,
                tiny_dnn::vec_t *dst) {
  tiny_dnn::image<float> resized = resize_image(img, width, height);

  tiny_dnn::image<> resized_uint8(resized);

  if (!mean.empty()) {
    tiny_dnn::image<float> normalized = subtract_scalar(resized, mean);
    *dst                              = normalized.to_vec();
  } else {
    *dst = resized.to_vec();
  }
}

std::vector<std::string> get_label_list(const std::string &label_file) {
  std::string line;
  std::ifstream ifs(label_file.c_str());

  if (ifs.fail() || ifs.bad()) {
    throw std::runtime_error("failed to open:" + label_file);
  }

  std::vector<std::string> lines;
  while (getline(ifs, line)) lines.push_back(line);

  return lines;
}

void test(const std::string &img_file) {
  auto labels = get_label_list("model/label.txt");
  //auto net    = tiny_dnn::create_net_from_caffe_prototxt(model_file);
  //std::cout << net->name() << std::endl;

  //tiny_dnn::reload_weight_from_caffe_protobinary(trained_file, net.get());
  auto net = std::make_shared<tiny_dnn::network<tiny_dnn::sequential>>("AlexNet");
  net->load("model/AlexNet_Model.bin", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::binary);

  // int channels = (*net)[0]->in_data_shape()[0].depth_;
  int width  = (*net)[0]->in_data_shape()[0].width_;
  int height = (*net)[0]->in_data_shape()[0].height_;

  auto mean = compute_mean(width, height);


  tiny_dnn::image<float> img(img_file, tiny_dnn::image_type::bgr);

  tiny_dnn::vec_t vec;

  preprocess(img, mean, width, height, &vec);

  clock_t begin = clock();

  auto result = net->predict(vec);

  clock_t end         = clock();
  double elapsed_secs = static_cast<double>(end - begin) / CLOCKS_PER_SEC;
  std::cout << "Elapsed time(s): " << elapsed_secs << std::endl;

  std::vector<tiny_dnn::float_t> sorted(result.begin(), result.end());

  int top_n = 5;
  partial_sort(sorted.begin(), sorted.begin() + top_n, sorted.end(),
               std::greater<tiny_dnn::float_t>());

  for (int i = 0; i < top_n; i++) {
    size_t idx =
      distance(result.begin(), find(result.begin(), result.end(), sorted[i]));
    std::cout << labels[idx] << "," << sorted[i] << std::endl;
  }
}

int main(int argc, char **argv) {
  int arg_channel          = 1;
  std::string img_file     = argv[arg_channel++];

  try {
    test(img_file);
  } catch (const tiny_dnn::nn_error &e) {
    std::cout << e.what() << std::endl;
  }
}

