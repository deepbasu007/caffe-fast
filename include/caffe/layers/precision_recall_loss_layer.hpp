#ifndef CAFFE_PRECISION_RECALL_LOSS_LAYER_HPP_
#define CAFFE_PRECISION_RECALL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

  /**
   * @brief compute area under the precision recall curve (AUPRC)
   * This layer needs a softmaxed prediction blob and a label blob as inputs.
   */
  template <typename Dtype>
  class PrecisionRecallLossLayer : public LossLayer<Dtype> {
   public:
    explicit PrecisionRecallLossLayer(const LayerParameter &param)
      : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                            const vector<Blob<Dtype>*> &top);
    virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                         const vector<Blob<Dtype>*> &top);

    virtual inline const char *type() const { return "PrecisionRecallLoss"; }

    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return -1; }
    virtual inline int MinTopBlobs() const { return 1; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                             const vector<Blob<Dtype>*> &top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*> &bottom,
                             const vector<Blob<Dtype>*> &top);
    virtual void Backward_cpu(
      const vector<Blob<Dtype>*> &top,
      const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom);
    virtual void Backward_gpu(
      const vector<Blob<Dtype>*> &top,
      const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom);

   public:
    Blob<Dtype> loss_;
  };

}  // namespace caffe

#endif  // CAFFE_PRECISION_RECALL_LOSS_LAYER_HPP_
