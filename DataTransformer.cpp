//
// Created by Michal Faber on 14/09/2017.
//
#include <fstream>
#include "DataTransformer.h"
#include "RNGen.h"


unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}


CPMDataTransformer::CPMDataTransformer(const TransformationParameter& param) : param_(param) {
  np_in_lmdb = param_.np_in_lmdb;
  np = param_.num_parts;
  is_table_set = false;
}

void CPMDataTransformer::InitRand() {
  const bool needs_rand = param_.mirror || param_.crop_size;
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new RNGen::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

int CPMDataTransformer::Rand(int n) {
  rng_t* rng =
      static_cast<rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

void CPMDataTransformer::SetAugTable(int numData){
  aug_degs.resize(numData);
  aug_flips.resize(numData);
  for(int i = 0; i < numData; i++){
    aug_degs[i].resize(param_.num_total_augs);
    aug_flips[i].resize(param_.num_total_augs);
  }
  //load table files
  char filename[100];
  sprintf(filename, "../../rotate_%d_%d.txt", param_.num_total_augs, numData);
  ifstream rot_file(filename);
  char filename2[100];
  sprintf(filename2, "../../flip_%d_%d.txt", param_.num_total_augs, numData);
  ifstream flip_file(filename2);

  for(int i = 0; i < numData; i++){
    for(int j = 0; j < param_.num_total_augs; j++){
      rot_file >> aug_degs[i][j];
      flip_file >> aug_flips[i][j];
    }
  }
}

void CPMDataTransformer::swapLeftRight(Joints& j) {
  if(np == 56){
    int right[8] = {3,4,5, 9,10,11,15,17};
    int left[8] =  {6,7,8,12,13,14,16,18};
    for(int i=0; i<8; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }

  else if(np == 43){
    int right[6] = {3,4,5,9,10,11};
    int left[6] = {6,7,8,12,13,14};
    for(int i=0; i<6; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
}

bool CPMDataTransformer::onPlane(Point p, Size img_size) {
  if(p.x < 0 || p.y < 0) return false;
  if(p.x >= img_size.width || p.y >= img_size.height) return false;
  return true;
}

void CPMDataTransformer::RotatePoint(Point2f& p, Mat R){
  Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}

void CPMDataTransformer::TransformMetaJoints(MetaData& meta) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  TransformJoints(meta.joint_self);
  for(int i=0;i<meta.joint_others.size();i++){
    TransformJoints(meta.joint_others[i]);
  }
}

void CPMDataTransformer::TransformJoints(Joints& j) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  Joints jo = j;

  if(np == 56){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
      jo.joints[i] = (j.joints[COCO_to_ours_1[i]-1] + j.joints[COCO_to_ours_2[i]-1]) * 0.5;
      if(j.isVisible[COCO_to_ours_1[i]-1]==2 || j.isVisible[COCO_to_ours_2[i]-1]==2){
        jo.isVisible[i] = 2;
      }
      else if(j.isVisible[COCO_to_ours_1[i]-1]==3 || j.isVisible[COCO_to_ours_2[i]-1]==3){
        jo.isVisible[i] = 3;
      }
      else {
        jo.isVisible[i] = j.isVisible[COCO_to_ours_1[i]-1] && j.isVisible[COCO_to_ours_2[i]-1];
      }
    }
  }

  else if(np == 43){
    int MPI_to_ours_1[15] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7};
    int MPI_to_ours_2[15] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 6};
    jo.joints.resize(np);
    jo.isVisible.resize(np);

    for(int i=0;i<15;i++){
      jo.joints[i] = (j.joints[MPI_to_ours_1[i]] + j.joints[MPI_to_ours_2[i]]) * 0.5;
      if(j.isVisible[MPI_to_ours_1[i]]==2 || j.isVisible[MPI_to_ours_2[i]]==2){
        jo.isVisible[i] = 2;
      }
      else {
        jo.isVisible[i] = j.isVisible[MPI_to_ours_1[i]] && j.isVisible[MPI_to_ours_2[i]];
      }
    }
  }

  j = jo;
}

bool CPMDataTransformer::augmentation_flip(Mat& img_src, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  bool doflip;
  if(param_.aug_way == "rand"){
    float dice = Rand(RAND_MAX) / static_cast <float> (RAND_MAX);
    doflip = (dice <= param_.flip_prob);
  }
  else if(param_.aug_way == "table"){
    doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs] == 1);
  }
  else {
    doflip = 0;
  }

  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;
    if(mode>4){
      flip(mask_miss, mask_miss, 1);
    }
    if(mode>5){
      flip(mask_all, mask_all, 1);
    }
    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i=0; i<np; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint)
      swapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i=0; i<np; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint)
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

float CPMDataTransformer::augmentation_rotate(Mat& img_src, Mat& img_dst, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {

  float degree;
  if(param_.aug_way == "rand"){
    float dice = Rand(RAND_MAX) / static_cast <float> (RAND_MAX);
    degree = (dice - 0.5) * 2 * param_.max_rotate_degree;
  }
  else if(param_.aug_way == "table"){
    degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs];
  }
  else {
    degree = 0;
  }

  Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;

  warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  if(mode >4){
    warpAffine(mask_miss, mask_miss, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255)); //Scalar(0) for MPI, COCO with Scalar(255);
  }
  if(mode >5){
    warpAffine(mask_all, mask_all, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));
  }

  //adjust meta data
  RotatePoint(meta.objpos, R);
  for(int i=0; i<np; i++){
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    RotatePoint(meta.objpos_other[p], R);
    for(int i=0; i<np; i++){
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}

float CPMDataTransformer::augmentation_scale(Mat& img_src, Mat& img_temp, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  float dice = Rand(RAND_MAX) / static_cast <float> (RAND_MAX);
  float scale_multiplier;

  if(dice > param_.scale_prob) {
    img_temp = img_src.clone();
    scale_multiplier = 1;
  }
  else {
    float dice2 = Rand(RAND_MAX) / static_cast <float> (RAND_MAX);
    scale_multiplier = (param_.scale_max - param_.scale_min) * dice2 + param_.scale_min; //linear shear into [scale_min, scale_max]
  }
  float scale_abs = param_.target_dist/meta.scale_self;
  float scale = scale_abs * scale_multiplier;

  resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);
  if(mode>4){
    resize(mask_miss, mask_miss, Size(), scale, scale, INTER_CUBIC);
  }
  if(mode>5){
    resize(mask_all, mask_all, Size(), scale, scale, INTER_CUBIC);
  }

  //modify meta data
  meta.objpos *= scale;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] *= scale;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] *= scale;
    }
  }
  return scale_multiplier;
}

Size CPMDataTransformer::augmentation_croppad(Mat& img_src, Mat& img_dst, Mat& mask_miss, Mat& mask_miss_aug, Mat& mask_all, Mat& mask_all_aug, MetaData& meta, int mode) {
  float dice_x = Rand(RAND_MAX) / static_cast <float> (RAND_MAX);
  float dice_y = Rand(RAND_MAX) / static_cast <float> (RAND_MAX);

  int crop_x = param_.crop_size_x;
  int crop_y = param_.crop_size_y;

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max);
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max);

  Point2i center = meta.objpos + Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));

  img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
  mask_miss_aug = Mat::zeros(crop_y, crop_x, CV_8UC1) + Scalar(255); //for MPI, COCO with Scalar(255);
  mask_all_aug = Mat::zeros(crop_y, crop_x, CV_8UC1);
  for(int i=0;i<crop_y;i++){
    for(int j=0;j<crop_x;j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
        img_dst.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
        if(mode>4){
          mask_miss_aug.at<uchar>(i,j) = mask_miss.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
        if(mode>5){
          mask_all_aug.at<uchar>(i,j) = mask_all.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
      }
    }
  }

  //modify meta data
  Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return Size(x_offset, y_offset);
}

void CPMDataTransformer::putGaussianMaps(double* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma){
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent > 4.6052){ //ln(100) = -ln(1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1)
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

void CPMDataTransformer::putVecMaps(double* entryX, double* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
  //int thre = 4;
  centerB = centerB*0.125;
  centerA = centerA*0.125;
  Point2f bc = centerB - centerA;
  int min_x = std::max( int(round(std::min(centerA.x, centerB.x)-thre)), 0);
  int max_x = std::min( int(round(std::max(centerA.x, centerB.x)+thre)), grid_x);

  int min_y = std::max( int(round(std::min(centerA.y, centerB.y)-thre)), 0);
  int max_y = std::min( int(round(std::max(centerA.y, centerB.y)+thre)), grid_y);

  float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;

  for (int g_y = min_y; g_y < max_y; g_y++){
    for (int g_x = min_x; g_x < max_x; g_x++){
      Point2f ba;
      ba.x = g_x - centerA.x;
      ba.y = g_y - centerA.y;
      float dist = std::abs(ba.x*bc.y -ba.y*bc.x);

      if(dist <= thre){
        int cnt = count.at<uchar>(g_y, g_x);
        if (cnt == 0){
          entryX[g_y*grid_x + g_x] = bc.x;
          entryY[g_y*grid_x + g_x] = bc.y;
        }
        else{
          // averaging when limbs of multiple persons overlap
          entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1);
          entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1);
          count.at<uchar>(g_y, g_x) = cnt + 1;
        }
      }

    }
  }
}

void CPMDataTransformer::generateLabelMap(double* transformed_label, Mat& img_aug, MetaData meta) {
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int stride = param_.stride;
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;
  int mode = 5; // TO DO: make this as a parameter

  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      for (int i = np+1; i < 2*(np+1); i++){
        if (mode == 6 && i == (2*np + 1))
          continue;
        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
      }
    }
  }

  if (np == 56){
    // add gausians for all parts
    for (int i = 0; i < 18; i++){
      Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+39)*channelOffset, center, param_.stride,
                        grid_x, grid_y, param_.sigma); //self
      }
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+39)*channelOffset, center, param_.stride,
                          grid_x, grid_y, param_.sigma);
        }
      }
    }

    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
    int thre = 1;

    for(int i=0;i<19;i++){
      Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = meta.joint_self;
      if(jo.isVisible[mid_1[i]-1]<=1 && jo.isVisible[mid_2[i]-1]<=1){
        //putVecPeaks
        putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset,
                   count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], param_.stride, grid_x, grid_y, param_.sigma, thre); //self
      }

      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Joints jo2 = meta.joint_others[j];
        if(jo2.isVisible[mid_1[i]-1]<=1 && jo2.isVisible[mid_2[i]-1]<=1){
          //putVecPeaks
          putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset,
                     count, jo2.joints[mid_1[i]-1], jo2.joints[mid_2[i]-1], param_.stride, grid_x, grid_y, param_.sigma, thre); //self
        }
      }
    }

    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        float maximum = 0;
        //second background channel
        for (int i = np+39; i < np+57; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
      }
    }
  }

  else if (np == 43){
    for (int i = 0; i < 15; i++){
      Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+29)*channelOffset, center, param_.stride,
                        grid_x, grid_y, param_.sigma); //self
      }
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+29)*channelOffset, center, param_.stride,
                          grid_x, grid_y, param_.sigma);
        }
      }
    }

    int mid_1[14] = {0, 1, 2, 3, 1, 5, 6, 1, 14, 8, 9,  14, 11, 12};
    int mid_2[14] = {1, 2, 3, 4, 5, 6, 7, 14, 8, 9, 10, 11, 12, 13};
    int thre = 1;

    for(int i=0;i<14;i++){
      Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = meta.joint_self;
      if(jo.isVisible[mid_1[i]]<=1 && jo.isVisible[mid_2[i]]<=1){
        putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset,
                   count, jo.joints[mid_1[i]], jo.joints[mid_2[i]], param_.stride, grid_x, grid_y, param_.sigma, thre); //self
      }

      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Joints jo2 = meta.joint_others[j];
        if(jo2.isVisible[mid_1[i]]<=1 && jo2.isVisible[mid_2[i]]<=1){
          putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset,
                     count, jo2.joints[mid_1[i]], jo2.joints[mid_2[i]], param_.stride, grid_x, grid_y, param_.sigma, thre); //self
        }
      }
    }

    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        float maximum = 0;
        //second background channel
        for (int i = np+29; i < np+44; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
      }
    }
  }

  //visualize
  if(1 && param_.visualize){
    Mat label_map;
    for(int i = 95; i < 2*(np+1); i++){
      label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
      for (int g_y = 0; g_y < grid_y; g_y++){
        for (int g_x = 0; g_x < grid_x; g_x++){
          label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[i*channelOffset + g_y*grid_x + g_x]*255);
        }
      }
      resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
      applyColorMap(label_map, label_map, COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

      char imagename [100];
      sprintf(imagename, "augment_%04d_label_part_%02d.jpg", meta.write_number, i);
      imwrite(imagename, label_map);
    }

  }
}

void CPMDataTransformer::clahe(Mat& bgr_image, int tileSize, int clipLimit) {
  Mat lab_image;
  cvtColor(bgr_image, lab_image, CV_BGR2Lab);

  // Extract the L channel
  vector<Mat> lab_planes(3);
  split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  // apply the CLAHE algorithm to the L channel
  Ptr<CLAHE> clahe = createCLAHE(clipLimit, Size(tileSize, tileSize));
  //clahe->setClipLimit(4);
  Mat dst;
  clahe->apply(lab_planes[0], dst);

  // Merge the the color planes back into an Lab image
  dst.copyTo(lab_planes[0]);
  merge(lab_planes, lab_image);

  // convert back to RGB
  Mat image_clahe;
  cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  bgr_image = image_clahe.clone();
}

void DecodeFloats(const uchar *data, size_t idx, float* pf, size_t len) {
  memcpy(pf, data + idx, len * sizeof(float));
}

string DecodeString(const uchar *data, size_t idx) {
  string result = "";
  int i = 0;
  while(data[idx+i] != 0){
    result.push_back(char(data[idx+i]));
    i++;
  }
  return result;
}

void CPMDataTransformer::ReadMetaData(MetaData& meta, const uchar *data, size_t offset3, size_t offset1) { //very specific to genLMDB.py
  // ------------------- Dataset name ----------------------
  meta.dataset = DecodeString(data, offset3);
  // ------------------- Image Dimension -------------------
  float height, width;
  DecodeFloats(data, offset3+offset1, &height, 1);
  DecodeFloats(data, offset3+offset1+4, &width, 1);
  meta.img_size = Size(width, height);
  // ----------- Validation, nop, counters -----------------
  meta.isValidation = (data[offset3+2*offset1]==0 ? false : true);
  meta.numOtherPeople = (int)data[offset3+2*offset1+1];
  meta.people_index = (int)data[offset3+2*offset1+2];
  float annolist_index;
  DecodeFloats(data, offset3+2*offset1+3, &annolist_index, 1);
  meta.annolist_index = (int)annolist_index;
  float write_number;
  DecodeFloats(data, offset3+2*offset1+7, &write_number, 1);
  meta.write_number = (int)write_number;
  float total_write_number;
  DecodeFloats(data, offset3+2*offset1+11, &total_write_number, 1);
  meta.total_write_number = (int)total_write_number;

  // count epochs according to counters
  static int cur_epoch = -1;
  if(meta.write_number == 0){
    cur_epoch++;
  }
  meta.epoch = cur_epoch;
  if(param_.aug_way == "table" && !is_table_set){
    SetAugTable(meta.total_write_number);
    is_table_set = true;
  }
  // ------------------- objpos -----------------------
  DecodeFloats(data, offset3+3*offset1, &meta.objpos.x, 1);
  DecodeFloats(data, offset3+3*offset1+4, &meta.objpos.y, 1);
  // ------------ scale_self, joint_self --------------
  DecodeFloats(data, offset3+4*offset1, &meta.scale_self, 1);
  meta.joint_self.joints.resize(np_in_lmdb);
  meta.joint_self.isVisible.resize(np_in_lmdb);
  for(int i=0; i<np_in_lmdb; i++){
    DecodeFloats(data, offset3+5*offset1+4*i, &meta.joint_self.joints[i].x, 1);
    DecodeFloats(data, offset3+6*offset1+4*i, &meta.joint_self.joints[i].y, 1);
    float isVisible;
    DecodeFloats(data, offset3+7*offset1+4*i, &isVisible, 1);
    if (isVisible == 2){
      meta.joint_self.isVisible[i] = 3;
    }
    else{
      meta.joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1;
      if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
         meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
        meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
      }
    }
  }

  //others (7 lines loaded)
  meta.objpos_other.resize(meta.numOtherPeople);
  meta.scale_other.resize(meta.numOtherPeople);
  meta.joint_others.resize(meta.numOtherPeople);
  for(int p=0; p<meta.numOtherPeople; p++){
    DecodeFloats(data, offset3+(8+p)*offset1, &meta.objpos_other[p].x, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+4, &meta.objpos_other[p].y, 1);
    DecodeFloats(data, offset3+(8+meta.numOtherPeople)*offset1+4*p, &meta.scale_other[p], 1);
  }
  //8 + numOtherPeople lines loaded
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.joint_others[p].joints.resize(np_in_lmdb);
    meta.joint_others[p].isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++){
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p)*offset1+4*i, &meta.joint_others[p].joints[i].x, 1);
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+1)*offset1+4*i, &meta.joint_others[p].joints[i].y, 1);
      float isVisible;
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+2)*offset1+4*i, &isVisible, 1);
      if (isVisible == 2){
        meta.joint_others[p].isVisible[i] = 3;
      }
      else {
        meta.joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
        if (meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 ||
            meta.joint_others[p].joints[i].x >= meta.img_size.width ||
            meta.joint_others[p].joints[i].y >= meta.img_size.height) {
          meta.joint_others[p].isVisible[i] = 2; // 2 means cropped, 1 means occluded by still on image
        }
      }
    }
  }
}

void CPMDataTransformer::dumpEverything(double* transformed_data, double* transformed_label, MetaData meta){

  char filename[100];
  sprintf(filename, "transformed_data_%04d_%02d", meta.annolist_index, meta.people_index);
  ofstream myfile;
  myfile.open(filename);
  int data_length = param_.crop_size_y * param_.crop_size_x * 4;

  for(int i = 0; i<data_length; i++){
    myfile << transformed_data[i] << " ";
  }
  myfile.close();

  sprintf(filename, "transformed_label_%04d_%02d", meta.annolist_index, meta.people_index);
  myfile.open(filename);
  int label_length = param_.crop_size_y * param_.crop_size_x / param_.stride / param_.stride * (param_.num_parts+1);
  for(int i = 0; i<label_length; i++){
    myfile << transformed_label[i] << " ";
  }
  myfile.close();
}

void CPMDataTransformer::Transform_nv(const uchar *data, const int datum_channels, const int datum_height, const int datum_width, uchar* transformed_data, double* transformed_label) {
  int clahe_tileSize = param_.clahe_tile_size;
  int clahe_clipLimit = param_.clahe_clip_limit;
  //float targetDist = 41.0/35.0;
  AugmentSelection as = {
      false,
      0.0,
      Size(),
      0,
  };
  MetaData meta;

  // To do: make this a parameter in caffe.proto
  const int mode = 5;  //related to datum.channels();
  int crop_x = param_.crop_size_x;
  int crop_y = param_.crop_size_y;

  //before any transformation, get the image from datum
  Mat img = Mat::zeros(datum_height, datum_width, CV_8UC3);
  Mat mask_all, mask_miss;
  if(mode >= 5){
    mask_miss = Mat::ones(datum_height, datum_width, CV_8UC1);
  }
  if(mode == 6){
    mask_all = Mat::zeros(datum_height, datum_width, CV_8UC1);
  }

  int offset = img.rows * img.cols;
  int dindex;
  uchar d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      Vec3b& rgb = img.at<Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;

        d_element = data[dindex];

        rgb[c] = d_element;
      }

      if(mode >= 5){
        dindex = 4*offset + i*img.cols + j;

        d_element = data[dindex];

        if (round(d_element/255)!=1 && round(d_element/255)!=0){
          cout << d_element << " " << round(d_element/255) << endl;
        }
        mask_miss.at<uchar>(i, j) = d_element; //round(d_element/255);
      }

      if(mode == 6){
        dindex = 5*offset + i*img.cols + j;

        d_element = data[dindex];

        mask_all.at<uchar>(i, j) = d_element;
      }
    }
  }

  //color, contract
  if(param_.do_clahe)
    clahe(img, clahe_tileSize, clahe_clipLimit);

  if(param_.gray == 1){
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }

  int offset3 = 3 * offset;
  int offset1 = datum_width;
  int stride = param_.stride;
  ReadMetaData(meta, data, offset3, offset1);
  if(param_.transform_body_joint) // we expect to transform body joints, and not to transform hand joints
    TransformMetaJoints(meta);

  //Start transforming
  Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
  Mat mask_miss_aug, mask_all_aug ;
  Mat img_temp, img_temp2, img_temp3; //size determined by scale
  as.scale = augmentation_scale(img, img_temp, mask_miss, mask_all, meta, mode);
  as.degree = augmentation_rotate(img_temp, img_temp2, mask_miss, mask_all, meta, mode);
  //if(0 && param_.visualize())
  //  visualize(img_temp2, meta, as);
  as.crop = augmentation_croppad(img_temp2, img_temp3, mask_miss, mask_miss_aug, mask_all, mask_all_aug, meta, mode);
  //if(0 && param_.visualize())
  //  visualize(img_temp3, meta, as);
  as.flip = augmentation_flip(img_temp3, img_aug, mask_miss_aug, mask_all_aug, meta, mode);
  //if(param_.visualize())
  //  visualize(img_aug, meta, as);

  if (mode > 4){
    resize(mask_miss_aug, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
  }
  if (mode > 5){
    resize(mask_all_aug, mask_all_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
  }

  //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
  offset = img_aug.rows * img_aug.cols;
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;

  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      Vec3b& rgb = img_aug.at<Vec3b>(i, j);
      transformed_data[0*offset + i*img_aug.cols + j] = rgb[0];
      transformed_data[1*offset + i*img_aug.cols + j] = rgb[1];
      transformed_data[2*offset + i*img_aug.cols + j] = rgb[2];
    }
  }

  // label size is image size/ stride
  if (mode > 4){
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        for (int i = 0; i < np; i++){
          float weight = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255; //mask_miss_aug.at<uchar>(i, j);
          if (meta.joint_self.isVisible[i] != 3){
            transformed_label[i*channelOffset + g_y*grid_x + g_x] = weight;
          }
        }
        // background channel
        if(mode == 5){
          transformed_label[np*channelOffset + g_y*grid_x + g_x] = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255;
        }
        if(mode > 5){
          transformed_label[np*channelOffset + g_y*grid_x + g_x] = 1;
          transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = float(mask_all_aug.at<uchar>(g_y, g_x)) /255;
        }
      }
    }
  }

  generateLabelMap(transformed_label, img_aug, meta);

  //starts to visualize everything (transformed_data in 4 ch, label) fed into conv1
  //if(param_.visualize()){
  //dumpEverything(transformed_data, transformed_label, meta);
  //}
}