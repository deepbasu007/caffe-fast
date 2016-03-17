#ifndef CAFFE_PATCH_TRANSFORMER_LAYER_HPP_
#define CAFFE_PATCH_TRANSFORMER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

  /**
   * @brief Perform some transformation to input data and label.
   */
  template <typename Dtype>
  class PatchTransformerLayer : public Layer<Dtype> {
  public:
    explicit PatchTransformerLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
    void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                    const vector<Blob<Dtype>*> &top);
    virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                         const vector<Blob<Dtype>*> &top);
    virtual inline const char* type() const { return "PatchTransformer"; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MaxBottomBlobs() const { return 2; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                             const vector<Blob<Dtype>*> &top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
                              const vector<bool> &propagate_down,
                              const vector<Blob<Dtype>*> &bottom) {
      NOT_IMPLEMENTED;
    }

  private:
    cv::Mat ConvertToCVMat(const Dtype *data, const int &channels,
                           const int &height, const int &width);
    void ConvertFromCVMat(const cv::Mat img, Dtype *data);
  };

}  // namespace caffe

#endif  // CAFFE_PATCH_TRANSFORMER_LAYER_HPP_
