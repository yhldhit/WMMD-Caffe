#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
void MMDLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  input_num_ = bottom[0]->count(0, 1);
  data_dim_ = bottom[0]->count(1);
  num_of_kernel_ = this->layer_param_.mmd_param().num_of_kernel();
  mmd_lambda_ = this->layer_param_.mmd_param().mmd_lambda();
  iter_of_epoch_ = this->layer_param_.mmd_param().iter_of_epoch();
  fix_gamma_ = this->layer_param_.mmd_param().fix_gamma();
  beta_ = new Dtype[num_of_kernel_];
  mmd_lr_ = this->layer_param_.mmd_param().mmd_lr();
  quad_weight_ = this->layer_param_.mmd_param().quad_weight();
  mmd_lock_ = this->layer_param_.mmd_param().mmd_lock();//specify when the weights of each layer need update
  num_class_ = this->layer_param_.mmd_param().num_class();
  test_inter_ = this->layer_param_.mmd_param().test_inter();
  total_iter_test_ = this->layer_param_.mmd_param().total_iter_test();
  //weig_ = new Dtype[10];
  sum_of_weig_ = new Dtype[num_class_];

  caffe_set(num_class_,Dtype(0),sum_of_weig_);
  //caffe_set(10,Dtype(1.0),weig_);
  caffe_set(num_of_kernel_, Dtype(1.0) / num_of_kernel_, beta_);
  now_iter_ = 0;
  now_iter_test_ = 0;

  sum_of_epoch_ = new Dtype[num_of_kernel_];
  caffe_set(num_of_kernel_, Dtype(0), sum_of_epoch_);
  gamma_ = Dtype(-1);
  Q_ = new Dtype* [num_of_kernel_];
  for(int i = 0; i < num_of_kernel_; i++){
      Q_[i] = new Dtype[num_of_kernel_];
      caffe_set(num_of_kernel_, Dtype(0), Q_[i]);
  }
  variance_ = new Dtype[num_of_kernel_];
  caffe_set(num_of_kernel_, Dtype(0), variance_);
  sum_of_pure_mmd_ = new Dtype[num_of_kernel_];
  caffe_set(num_of_kernel_, Dtype(0), sum_of_pure_mmd_);
  all_sample_num_ = 0;
  total_target_num = 0;
  kernel_mul_ = this->layer_param_.mmd_param().kernel_mul();
  if(this->layer_param_.mmd_param().method() == "max"){
        method_number_ = 1;
        top_k_ = this->layer_param_.mmd_param().method_param().top_num();
  }
  else if(this->layer_param_.mmd_param().method() == "none"){
        method_number_ = 0;
  }
  else if(this->layer_param_.mmd_param().method() == "L2"){
        method_number_ = 4;
        top_k_ = this->layer_param_.mmd_param().method_param().top_num();
        I_lambda_ = this->layer_param_.mmd_param().method_param().i_lambda();
  }
  else if(this->layer_param_.mmd_param().method() == "max_ratio"){
        top_k_ = this->layer_param_.mmd_param().method_param().top_num();
        method_number_ = 3;
  }
  LOG(INFO) << this->layer_param_.mmd_param().method() << " num: " << method_number_;
  source_index_ = new int[input_num_];
  target_index_ = new int[input_num_];

  mmd_data_.Reshape(1, 1, 1, data_dim_);

  class_num = bottom[2]->count(0);

  count_soft = new Dtype[class_num];
  avg_entropy = new Dtype[class_num];
  count_tmp = new int [class_num];
  caffe_set(class_num,Dtype(0.0),count_soft);
  caffe_set(class_num,Dtype(0.0),avg_entropy);
  count_hard = new int[class_num];
  caffe_set(class_num,0,count_hard);
  caffe_set (class_num,0,count_tmp);
  source_num_batch = new int[class_num];
  target_num_batch = new int[class_num];
  source_num_resamp = new int[class_num];
  target_num_resamp= new int[class_num];
  caffe_set(class_num,0,source_num_batch);
  caffe_set(class_num,0,target_num_batch);
  caffe_set(class_num,0,source_num_resamp);
  caffe_set(class_num,0,target_num_resamp);
  cross_entropy = 0.0;
  entropy_stand = 1.0;//for all class
  entropy_thresh_ = this->layer_param_.mmd_param().entropy_thresh();//for every sample

}

template <typename Dtype>
void MMDLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MMDLossLayer);
#endif

INSTANTIATE_CLASS(MMDLossLayer);
REGISTER_LAYER_CLASS(MMDLoss);

}

