#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
	const int label_tmp = static_cast<int>(label[(n * spatial_dim + s)*2]);
    const int label_value = static_cast<int>(label[(n * spatial_dim + s) * 2 + 1]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      
      /*loss[index] = 0;
      counts[index] = 0;*/
	  
	  loss[index] = -log(max(prob_data[n * dim + label_tmp * spatial_dim + s],
                      Dtype(FLT_MIN)));
	  counts[index] = 1;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* label_ = bottom[1]->cpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_; 
  //LOG(INFO)<<"this is forward for softmax";	
  //if this is test phrase 
  const Dtype* label_1 = bottom[1]->cpu_data();
  int flag = 0;
  for (int i = 0; i<outer_num_; i++){
      if (label_1[i*2]<0){
          //source data,train phrase
          flag = 1;
          iter=0;
          if(begin == 2){
              begin =0;
          }
          break;
      }
  }

  /*if (outer_num_ == batch_size){
      iter = iter;
  }*/
  if(flag == 0 && begin != 1){
      // test phrase,write data
      int sample_index = iter*batch_size + num_of_source;
      const Dtype* prob_top = bottom[0]->cpu_data();
      Dtype * prob_samp = new Dtype[dim];
      const char* trainfile2 = trainfile.c_str();
      const char* tmpfile2 = tmpfile.c_str();
      for (int i = 0; i < outer_num_; i++){
          caffe_copy(dim, prob_top + dim * i, prob_samp);
          int tclass = 0;
          for(int k = 1; k < dim; k++){
              if(prob_samp[k]>prob_samp[tclass]){
                  tclass = k;
              }
          }
              
          char path[100];
          int label_1;
          int label_2;

          //file for writting and reading
          FILE* fp_r = fopen(trainfile2,"r+");
          FILE* fp_w = fopen(tmpfile2,"a+");
          if(fp_r == NULL || fp_w ==NULL){
            LOG(ERROR)<<"file not exsit in softmax";
           };
              
	      int count = 0;
	      while(!feof(fp_r)){
              fscanf(fp_r,"%s %d %d",path,&label_1,&label_2);
              if (sample_index == num_of_source){
                  if(count< num_of_source){
                      fprintf(fp_w,"%s %d %d \n",path,label_1,label_2);
                  }
              }
              
              if(count == sample_index){
                  fprintf(fp_w,"%s %d %d \n",path,tclass,label_2);
                  break;
              }
              count ++;
          }
          fclose(fp_r);
          fclose(fp_w);
          sample_index++;
          if (sample_index == num_of_target + num_of_source)
            break;
      }
      //LOG(INFO)<<"write file is done";

	  if (sample_index == (num_of_target + num_of_source)){
          iter = 0;
          if(begin == 2) begin =0;
          remove(trainfile2);
          rename(tmpfile2,trainfile2);
      }
      else
          iter++;
  }
  else if(flag == 0 && begin == 1){
      if(iter == num_batch) {
          begin = 0; 
          iter =0;
      }
      else iter++;
  }

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(nthreads, counts, &count);
    loss /= (count > 0 ? count : Dtype(1));
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts,float target_lambda) {
  const int channels = dim / spatial_dim;
  //LOG(INFO) << "channels"<<channels;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    //const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int label_value = static_cast<int>(label[(n * spatial_dim + s) * 2 + 1]);
	const int label_tmp = static_cast<int>(label[n * spatial_dim + s]*2);
    if (has_ignore_label_ && label_value == ignore_label_) {
	 /* bottom_diff[n * dim + label_tmp * spatial_dim + s] -= 1;
		counts[index] = 1;*/
     for (int c = 0; c < channels; ++c) {
        if(c == label_tmp){
            bottom_diff[n * dim + c * spatial_dim + s] = target_lambda*bottom_diff[n * dim + c * spatial_dim + s] - target_lambda;
        }
        else
            bottom_diff[n * dim + c * spatial_dim + s] = target_lambda *bottom_diff[n*dim+c*spatial_dim+s];
      }
      counts[index] = target_lambda;
      //counts[index] = 1;
    }
    else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if(begin == 1) begin =2;
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,target_lambda);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / (count > 0 ? count : Dtype(1)), bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
