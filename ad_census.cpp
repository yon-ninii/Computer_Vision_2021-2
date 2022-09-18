#include <iostream>
#include <stdlib.h>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <cmath>
#include <algorithm>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

#define KSIZE 5
#define MAX_DISPARITY 16
#define rc 14
#define rp 14

float adaptive_w(float sim, float prox);
float prox_w(float p_x, float p_y, float q_x, float q_y);
float sim_w(float p, float q);
float SAD(float q, float q_);
Mat sliding_window(Mat& r_img, int c, int r, int rad);
int local_stereo_matching(Mat& l_img, Mat& r_img, int c, int r, int rad);

int main() {
   Mat l_img = imread("left.png", 0);
   Mat r_img = imread("right.png", 0);
   int hg = l_img.rows; int wh = l_img.cols;
   cout << wh << " " << hg << endl;
   int rad = KSIZE / 2;
   //cout << rad << endl;
   vector<float> disparities;
   Mat disparity_map = Mat::zeros(hg, wh, CV_32FC1);
   cout << disparity_map.rows;
   // pixel indexing
   int c, r;
   cout << "계산중...." << endl;
   for (r = rad; r < hg - rad; r++) {
      for (c = rad; c < wh - rad; c++) {
         //cout << "x : " << c << " y : " << r << " rad : " << rad << endl;
         disparity_map.at<float>(r, c) = local_stereo_matching(l_img, r_img, r, c, rad);
         //disparities.push_back(local_stereo_matching(l_img, r_img, r, c, rad));
      }
   }
   //cout << "sizeof disparities " << disparities.size() << endl;
   //Mat dis_map = Mat(disparities, CV_32FC1);
   //// 열이 3인 matrix로 변환
   //dis_map = dis_map.reshape(0, wh);

   //imshow("disparityyyyyyyyy", dis_map);
   //waitKey();
   imshow("disparity", disparity_map);
   waitKey();
   destroyAllWindows();
}
float adaptive_w(float sim, float prox) {
   return exp(-(sim / rc + prox / rp));
}
float prox_w(float p_x, float p_y, float q_x, float q_y) {
   return hypot(p_x - q_x, p_y - q_y);
}
float sim_w(float p, float q) {
   return abs(p - q);
}
float SAD(float q, float q_) {
   return abs(q - q_);
}

int local_stereo_matching(Mat& l_img, Mat& r_img, int r, int c, int rad) {
   vector<float> l_pixels;
   vector<float> r_pixels;
   vector<float> costs; // cost vector
   float u_cost = 0, l_cost = 0, cost = 0, min_cost = 0;
   //cout << c << " x " << r << endl;
   // left image kernel pixel
   int cnt = 0;
   for (int y = r - rad; y <= r + rad; y++) {
      for (int x = c - rad; x <= c + rad; x++) {
         cnt++;
         l_pixels.push_back(l_img.at<float>(y, x));
      }
   }
   //cout << "cnt : " << cnt << endl;
   Mat l_pixel = Mat(l_pixels, CV_32FC1);
   // 열이 3인 matrix로 변환
   l_pixel = l_pixel.reshape(0, KSIZE);
   int rimg_col_l = c - MAX_DISPARITY / 2;
   int rimg_col_r = c + MAX_DISPARITY / 2;

   if (rimg_col_l <= 0) {
      rimg_col_l = rad;
   }
   if (rimg_col_r >= r_img.cols) {
      rimg_col_r = r_img.cols - 1;
   }
   int idx = 0;
   //cout << rimg_col_l << " ~ " << rimg_col_r << "diff : " << rimg_col_r - rimg_col_l << endl;
   // right image kernel pixel and compare with left image kernel pixel
   int p_idx = ((KSIZE * KSIZE) / 2);
   int p_cnt = 0;
   //int cntcnt = 0;
   //cout << p_idx << endl;
   for (rimg_col_l; rimg_col_l <= rimg_col_r; rimg_col_l++) {
      for (int y = r - rad; y <= r + rad; y++) {
         for (int x = rimg_col_l - rad; x <= rimg_col_l + rad; x++) {
            //cntcnt++;

            //cout << x << ", " << y << "// rimgcol_l : " << rimg_col_l << " rimgcol_r : " << rimg_col_r << "  r_img_cols " << r_img.cols <<endl;
            float pw = prox_w(rad, rad, x, y);
            float l_w = adaptive_w(sim_w(l_pixels[p_idx], l_pixels[p_cnt]), pw);
            /*if (p_idx == p_cnt) {
               cout << l_pixels[p_idx] << " " <<l_pixels[p_cnt] << endl;
            }*/
            //cout << p_idx << " " << p_cnt << endl;
            float r_w = adaptive_w(sim_w(r_img.at<float>(y, rimg_col_l), r_img.at<float>(y, x)), pw);
            /*float pw = prox_w(rad, rad, x, y);
            float l_w = adaptive_w(sim_w(l_pixel.at<float>(rad, rad), l_pixel.at<float>(y, x))
               , pw);
            float r_w = adaptive_w(sim_w(r_img.at<float>(rad, rad), r_img.at<float>(y, x))
               , pw);*/
            float lr_w = l_w * r_w;
            u_cost += lr_w * SAD(l_pixels[p_cnt++], r_img.at<float>(y, x));
            l_cost += lr_w;
         }
      }
      //cout << "연산횟수 : " << cntcnt << endl;
      p_cnt = 0;
      cost = u_cost / l_cost;
      if (costs.size() == 0) {
         costs.push_back(cost);
      }
   }
}