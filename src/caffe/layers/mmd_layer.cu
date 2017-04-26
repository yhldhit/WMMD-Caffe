#include <algorithm>
#include <cfloat>
#include <vector>
#include <CGAL/basic.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

#include <CGAL/MP_Float.h>
#include <cmath>
#include <float.h>
typedef CGAL::MP_Float ET;

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/output_matrix.hpp"
//This file is to esitmate the weight based on prediction on target domain

typedef CGAL::Quadratic_program_from_iterators
<float **,float*,CGAL::Const_oneset_iterator<CGAL::Comparison_result>,
    bool*, float*,bool*,float*,float**,float*> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

namespace caffe {

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& top,
const vector<Blob<Dtype>*>& bottom){
}

template <typename Dtype>
void perm_source_and_target(int num, int* source_index, int* target_index,
        int& size_of_source, int& size_of_target, const Dtype* label){
    int source_pos = 0;
    int target_pos = 0;
    for(int i = 0;i < num;++i){
        if(label[i * 2] < 0){
            //source data
            source_index[source_pos++] = i;
        }
        else{
            //target data
            target_index[target_pos++] = i;
        }
    }
    size_of_source = source_pos;
    size_of_target = target_pos;
}

template <typename Dtype>
std::vector<std::pair<Dtype, int> > maxn(int num_of_max, Dtype* mmd, int num_of_kernel){
    std::vector<std::pair<Dtype, int> > temp;
    for(int i = 0; i < num_of_kernel; i++){
        temp.push_back(std::make_pair(mmd[i], i));
    }
    std::partial_sort(
            temp.begin(), temp.begin() + num_of_max, temp.end(), std::greater<std::pair<Dtype, int> >());
    return temp;
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if(mmd_lambda_ == 0){
        return;
    }
    LOG(INFO)<<"*************learn mmd based on prediction on target domain*****************";
    now_iter_++;
    Dtype sum;

    //const int class_num = bottom[2]->count(0);

    /***********for amazon***************/
    //const int num_source_sam[31] = {92,82,72,82,36,94,91,97,97,81,99,100,100,98,100,99,100,94,96,95,93,100,98,98,90,75,100,99,99,96,64};
    //const int source_sam = 2817;

    /***********for webcam*************/
    //const int num_target_sam[31] = {29,21,28,12,16,31,40,18,21,19,27,27,30,19,30,43,30,27,28,32,16,20,30,27,40,11,25,30,24,23,21};
    //const int target_sam = 795;

    /***********for dslr***************/
    //const int num_target_sam[31] = {12,21,24,12,16,12,13,14,15,15,13,10,24,16,31,22,12,8,10,10,13,15,23,18,10,7,18,26,21,22,15};
    //const int target_sam = 498;

    /*******for amazon********/
    //const int num_target_sam[10] = {92,82,94,99,100,100,99,100,94,98};
    //const int target_sam = 958;

    /*******for webcam*******************/
    //const int num_target_sam[10] = {29,21,31,27,27,30,43,30,27,30};
    //const int target_sam = 295;

    /********for dslr**************/
    const int num_source_sam[10] = {12,21,12,13,10,24,22,12,8,23};
    const int source_sam = 157;

    /*******for caltech256*********/
    const int num_target_sam[10] = {151,110,100,138,85,128,133,94,87,97};
    const int target_sam = 1123;

    /********for voc12**********/
    //const int num_target_sam[12] = {670,552,765,508,706,421,1161,1286,482,575,526,4087};
    //const int target_sam = 11739;

    /********for bing**********/
    //const int num_source_sam[12] = {446,480,446,484,467,518,511,496,463,445,471,489};
    //const int source_sam = 5716;
    
    float *ratio_target = new float [class_num];
    float *ratio_source = new float [class_num];
    float *weight_gd = new float[class_num];

    for (int i= 0;i<class_num;i++){
        ratio_target[i] = num_target_sam[i]*1.0/target_sam;
        ratio_source[i] = num_source_sam[i]*1.0/source_sam;
        //weight_gd[i] = ratio_target[i]/ratio_source[i];
        weight_gd[i] = ratio_target[i];
    }



    float weightSam[input_num_];
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* weig_ = bottom[2]->cpu_data();
    //Dtype* weig_diff_ = new Dtype[class_num];
    //caffe_set(class_num,Dtype(0.0),weig_diff_);
    //caffe_add(10,weig_diff_,sum_of_weig_,sum_of_weig_);
    //assign sample with weights
    for (int i =0;i<input_num_;i++){
        if (label[2*i]==-1){
            weightSam[i] = weig_[int(label[2*i+1])]/ratio_source[int(label[2*i+1])];
            //source_num_batch[int(label[2*i+1])] +=1;
        }
        else{
            weightSam[i] = 1.0;
            //target_num_batch[int(label[2*i])] +=1;
        }
    }

    caffe_gpu_asum(input_num_ * data_dim_, bottom[0]->gpu_diff(), &sum);
    LOG(INFO) << "before mmd diff " << sum;
    perm_source_and_target<Dtype>(input_num_, source_index_, target_index_,
            size_of_source_, size_of_target_, bottom[1]->cpu_data());
    int sample_num;
    if (size_of_source_ <= 1 || size_of_target_ <= 1){
        return;
    }
    if(size_of_source_ > size_of_target_){
        sample_num = size_of_source_;
    }
    else{
        //sample_num = size_of_source_;
        sample_num = size_of_target_;
    }
    int s1,s2,t1,t2;
    srand((unsigned int)time(0));
    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* tempX1 = mmd_data_.mutable_gpu_data();
    Dtype* tempX2 = mmd_data_.mutable_gpu_diff();

    Dtype square_distance;
    Dtype bandwidth = 0;
    for(int i = 0; i < input_num_; i++){
        //calculate the samples in the batch

        s1 = rand() % input_num_;
        s2 = rand() % input_num_;
        s2 = (s1 != s2) ? s2 : (s2 + 1) % input_num_;
        caffe_gpu_memcpy(sizeof(Dtype) * data_dim_, bottom_data + s1 * data_dim_, tempX1);
        caffe_gpu_memcpy(sizeof(Dtype) * data_dim_, bottom_data + s2 * data_dim_, tempX2);
        caffe_gpu_sub<Dtype>(data_dim_, tempX1, tempX2, tempX2);
        caffe_gpu_dot<Dtype>(data_dim_, tempX2, tempX2, &square_distance);
        bandwidth += square_distance;
    }
    if(fix_gamma_){
        gamma_ = gamma_ < 0 ? (Dtype)input_num_ / bandwidth : gamma_;
    }
    else{
        gamma_ = (Dtype)input_num_ / bandwidth;
    }
    LOG(INFO) << "bandwidth " << gamma_;
    Dtype loss = 0;

    Dtype* temp_loss1 = new Dtype[num_of_kernel_];
    Dtype* temp_loss2 = new Dtype[num_of_kernel_];
    Dtype* temp_loss3 = new Dtype[num_of_kernel_];
    Dtype* temp_loss4 = new Dtype[num_of_kernel_];

    //all_sample_num_ += sample_num;
    if (mmd_lock_ == 1 ){
        LOG(INFO)<<"update the net parameter";

        if (now_iter_ >= 300){//change the lock
            LOG(INFO)<< "now_iter_ 320";
            mmd_lock_ = 1-mmd_lock_;
            now_iter_ = 0;
        }

        for(int i = 0; i < sample_num; i++){
            //random get sample, insert code
            s1 = rand() % size_of_source_;
            s2 = rand() % size_of_source_;
            s2 = (s1 != s2) ? s2 : (s2 + 1) % size_of_source_;

            t1 = rand() % size_of_target_;
            t2 = rand() % size_of_target_;
            t2 = (t1 != t2) ? t2 : (t2 + 1) % size_of_target_;

            s1 = source_index_[s1];//the index from source domain from 0-batchsize
            s2 = source_index_[s2];
            t1 = target_index_[t1];
            t2 = target_index_[t2];
            //////////////
            Dtype square_sum = 0;
            Dtype factor_for_diff = 0;
            const Dtype* x_s1 = bottom_data + s1 * data_dim_;
            const Dtype* x_s2 = bottom_data + s2 * data_dim_;
            const Dtype* x_t1 = bottom_data + t1 * data_dim_;
            const Dtype* x_t2 = bottom_data + t2 * data_dim_;

            float weightSam1 = weightSam[s1];
            float weightSam2 = weightSam[s2];

            caffe_gpu_sub<Dtype>(data_dim_, x_s1, x_s2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_s2, x_s1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
            Dtype times = pow(kernel_mul_, (Dtype)(num_of_kernel_ / 2));
            Dtype temp_gamma = gamma_ / times;
            //Dtype quad1 = 0.0;
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n);

                sum_of_pure_mmd_[j] += temp_n;
                temp_n = temp_n * beta_[j];//beta_u*k_u
                //quad1 = quad1 + temp_n;
                if(i % 2 == 0){
                    temp_loss1[j] = temp_n;
                }
                else{
                    temp_loss2[j] = temp_n;
                }
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, weightSam1*weightSam2*mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, weightSam1*weightSam2*mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + s1 * data_dim_, bottom_diff + s1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + s2 * data_dim_, bottom_diff + s2 * data_dim_);

            factor_for_diff = 0;
            caffe_gpu_sub<Dtype>(data_dim_, x_s1, x_t2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_t2, x_s1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);//only for Gaussian kernel
            temp_gamma = gamma_ / times;
            //Dtype quad2 = 0.0;
            //************updata top[2]->mutable_cpu_data()
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n) * Dtype(-1);//Gaussian kernel

                sum_of_pure_mmd_[j] += temp_n;
                if(i % 2 == 0){
                    temp_loss1[j] += temp_n;
                }
                else{
                    temp_loss2[j] += temp_n;
                }
                temp_n = temp_n * beta_[j];
                //quad2 = quad2+temp_n;
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, weightSam1 * 1 * mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, weightSam1 * 1 * mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + s1 * data_dim_, bottom_diff + s1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + t2 * data_dim_, bottom_diff + t2 * data_dim_);


            factor_for_diff = 0;
            caffe_gpu_sub<Dtype>(data_dim_, x_t1, x_s2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_s2, x_t1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
            temp_gamma = gamma_ / times;
            //Dtype quad3 = 0.0;
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n) * Dtype(-1);

                sum_of_pure_mmd_[j] += temp_n;
                if(i % 2 == 0){
                    temp_loss1[j] += temp_n;
                }
                else{
                    temp_loss2[j] += temp_n;
                }
                temp_n = temp_n * beta_[j];
                //quad3 = quad3+temp_n;
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, weightSam2 * 1 * mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, weightSam2 * 1 * mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + t1 * data_dim_, bottom_diff + t1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + s2 * data_dim_, bottom_diff + s2 * data_dim_);

            factor_for_diff = 0;
            caffe_gpu_sub<Dtype>(data_dim_, x_t1, x_t2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_t2, x_t1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
            temp_gamma = gamma_ / times;
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n);

                sum_of_pure_mmd_[j] += temp_n;
                if(i % 2 == 0){
                    temp_loss1[j] += temp_n;
                }
                else{
                    temp_loss2[j] += temp_n;
                }
                temp_n = temp_n * beta_[j];
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + t1 * data_dim_, bottom_diff + t1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + t2 * data_dim_, bottom_diff + t2 * data_dim_);

        }
        delete [] temp_loss1;
        delete [] temp_loss2;
        delete [] temp_loss3;
        delete [] temp_loss4;
    }

    else if (mmd_lock_ == 0){//update the class weights
        caffe_gpu_set(input_num_*data_dim_, Dtype(0.0), bottom_diff);
        total_target_num += size_of_target_;
        if(data_dim_ == class_num){
            //all_sample_num_ += size_of_target_;
            //Dtype square_sum = 0;
            //soft label
			const Dtype* bottom_data_ = bottom[0]->cpu_data();
            int batch_index_ ;
            int label_index_ ;
            for (int i = 0; i < size_of_target_; i++){
                batch_index_ = target_index_[i];
                Dtype* prob_x_ = new Dtype[class_num];
                const Dtype* x_ = bottom_data_ + batch_index_ * data_dim_;
                caffe_copy(data_dim_,bottom_data_+batch_index_*data_dim_,prob_x_);
                Dtype tmp_entropy  = Dtype(0.0);
                if(true){
                    //all_sample_num_ ++;

                    //find out the max
                    label_index_ = 0;
                    for (int j = 0; j < data_dim_; j++){
                        if (x_[j] >x_[label_index_]){
                            label_index_ = j;
                        }
                    }
                    float max_prob = prob_x_[label_index_];
                    //exponent function
                    for (int j = 0; j < data_dim_; j++){
                        prob_x_[j] -= max_prob;
                        prob_x_[j] = (exp(prob_x_[j])==0)?1e-10:exp(prob_x_[j]);
                    }

                    float prob_sum = 0.0;
                    for (int j = 0; j < data_dim_; j++){
                        prob_sum += prob_x_[j];
                    }

                    for (int j = 0; j < data_dim_; j++){
                        prob_x_[j] /= prob_sum;
                        //count_soft[j] += prob_x_[j];
                        cross_entropy -= prob_x_[j] * log(prob_x_[j]);
                        tmp_entropy -= prob_x_[j]*log(prob_x_[j]);
                    }
                    if(tmp_entropy < entropy_thresh_){
                        all_sample_num_ ++;
                        count_hard[label_index_] ++ ;
                        for(int j = 0; j < data_dim_; j++){
                            count_soft[j] += prob_x_[j];
                        }
                        count_tmp [int(label[2*batch_index_])]++;
                        avg_entropy[label_index_] += tmp_entropy;
                    }
                    //count_hard[label_index_]++;
                }
                else
                    break;
            }
			//delete [] bottom_data_;


            if (total_target_num  >= 1000){//for layer fc8
                LOG(INFO)<<"update the class weights";
                cross_entropy /= total_target_num;
                Dtype* bottom_weight_ = bottom[2]->mutable_cpu_data();
                float* weight_soft = new float [class_num];
                float* weight_tmp = new float [class_num];
                float* weight_hard = new float [class_num];
                float square_soft = 0.0;
                float square_hard = 0.0;
                for (int k = 0; k<class_num; k++){
                    weight_tmp[k] = count_tmp[k]*1.0/all_sample_num_;
                    weight_soft[k] = count_soft[k]*1.0/all_sample_num_;
                    weight_hard[k] = count_hard[k]*1.0/all_sample_num_;
                    //if(cross_entropy >= entropy_stand){
                    bottom_weight_[k] = weight_hard[k];
                    //bottom_weight_[k] = weight_soft[k];
                    //}
                    avg_entropy[k] = avg_entropy[k]/count_hard[k];

                    LOG(INFO)<<"i:"<<k<<"\tweight_gd:"<<ratio_target[k]<<"\tweight_tmp:"<<weight_tmp[k]<<"\tweight_sr:"<<ratio_source[k];
                    LOG(INFO)<<"\tweight_soft:"<<weight_soft[k]<<"\tweight_hard:"<<weight_hard[k]<<"\tavg_entropy:"<<avg_entropy[k];
                    square_soft += (weight_soft[k]-weight_tmp[k])*(weight_soft[k]-weight_tmp[k]);
                    square_hard += (weight_hard[k]-weight_tmp[k])*(weight_hard[k]-weight_tmp[k]);
                    count_soft[k] = 0;
                    count_tmp[k] = 0;
                    count_hard[k] = 0;
                    avg_entropy[k] = 0;
                }
                /*
                if (square_soft < 0.001){
                    LOG(INFO)<<"update weight";
                    for (int i = 0; i<class_num; i++){
                        bottom_weight_[i] = weight_soft[i];
                    }
                }*/

                //adaptively update weight


                /*int count_cla = class_num;
                float residul_ = 1.0;
                Dtype *flag_ = new Dtype [class_num];
                caffe_set(class_num,Dtype(1),flag_);
                Dtype avg_weight = 0.0;
                for (int k = 0; k < class_num; k++){
                    if (avg_entropy[k] < 0.2){
                        count_cla--;
                        bottom_weight_[k] = weight_hard[k];
                        flag_[k] = 0;
                        residul_ -= weight_hard[k];
                    }
                    avg_entropy[k] = 0;
                }
                if (count_cla>0){
                    avg_weight = Dtype(residul_ / count_cla);
                    for (int k = 0; k < class_num; k++){
                        if(flag_[k]==1){
                            bottom_weight_[k] = avg_weight;
                        }
                    }

                }*/


                /*for(int k = 0; k < class_num; k++){
                    LOG(INFO)<< k << "\tlearned weight:\t" << bottom_weight_[k];
                }*/


                entropy_stand = entropy_stand>cross_entropy?entropy_stand:cross_entropy;
                LOG(INFO)<<"test sample number:\t"<<all_sample_num_;
                LOG(INFO) <<"soft distance:"<<square_soft<<"\thard distance:"<<square_hard<<"\tcross_entropy:"<<cross_entropy;
                //delete [] weight_soft;
                //delete [] weight_tmp;
                //delete [] weight_hard;
                //delete [] bottom_weight_;
                //now_iter_ =  0 ;
                //mmd_lock_ = 1-mmd_lock_;
            }
        }
        //change the lock
        if (total_target_num  >= 1000){//for layer fc7
            now_iter_ = 0;
            total_target_num = 0;
            all_sample_num_ = 0;
            mmd_lock_ = 1 - mmd_lock_;
            cross_entropy = 0;
        }

    }
    caffe_set(num_of_kernel_, Dtype(0), sum_of_epoch_);

    caffe_gpu_asum(input_num_ * data_dim_, bottom[0]->gpu_diff(), &sum);
    LOG(INFO) << "after mmd diff sum " << sum;
    LOG(INFO) << "------";
}

INSTANTIATE_LAYER_GPU_FUNCS(MMDLossLayer);

}


