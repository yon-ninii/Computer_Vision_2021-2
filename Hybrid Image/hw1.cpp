#include <iostream>
#include <vector>
#include <algorithm>

#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp" // opencv에서 필요한 헤더 파일 불러오기

using namespace std;
using namespace cv; // cv와 std namespace 쓰기

int N1 = 2; // 전역변수 N1
int N2 = 5; // 전역변수 N2

int conv3x3_Func(uchar* arr, int a, int b, int width, int height) { // 3x3 컨볼루션 함수
    //a, b : rows, columns 
    int kernel[3][3] = { 1,2,1,
                         2,4,2,
                         1,2,1 };    // 3x3 마스크 값 넣어주기

    int sum = 0;
    int sumKernel = 0;

    for (int j = -1; j < 2; j++) {
        for (int i = -1; i < 2; i++) { // 2중 for문
            if ((b + j) >= 0 && (b + j) < height && (a + i) >= 0 && (a + i) < width) { // i,j번째 픽셀이 영상의 총 크기 내부에 있을 때
                sum += arr[(b + j) * width + (a + i)] * kernel[i + 1][j + 1];          // 컨볼루션 연산 진행
                sumKernel += kernel[i + 1][j + 1];
            } 
        }
    }

    if (sumKernel == 0) { return sum; }
    else { return (sum / sumKernel); }
}

Mat GaussianFilter_Func(Mat Image) { // gaussian filter를 수행하는 함수

    int width = Image.cols;
    int height = Image.rows;

    Mat Final_Img(Image.size(), CV_8UC1);

    uchar* Prev_Data = Image.data;
    uchar* Final_Data = Final_Img.data;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) { // 2중 for문
            Final_Data[j * width + i] = conv3x3_Func(Prev_Data, i, j, width, height); // 3x3 컨볼루션 진행해주기
        }
    }

    return Final_Img;
}

Mat Sampling_Func(Mat Image) { // 샘플링 함수

    int width = Image.cols / 2; // 폭 절반으로 줄여주기
    int height = Image.rows / 2; // 높이 절반으로 줄여주기

    Mat dstImg(height, width, CV_8UC1);

    uchar* srcData = Image.data;
    uchar* dstData = dstImg.data;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            dstData[j * width + i] = srcData[(j * 2) * (Image.cols) + (i * 2)]; // 값 한칸씩 띄워서 넣어주기
        }
    }

    return dstImg;
}

vector<Mat> GaussianPyramid_Func(Mat Image) { // Gaussian Pyramid를 만들어주는 Function

    vector<Mat> Gaussian_Vector; // 마지막에 return해줄 모든 영상을 저장할 벡터 매트릭스 선언

    Gaussian_Vector.push_back(Image);

    for (int i = 0; i < 8; i++) { // 8 계층
        Image = Sampling_Func(Image); // 영상 샘플링
        Image = GaussianFilter_Func(Image); // Gaussian filter 적용

        Gaussian_Vector.push_back(Image);
    }
    return Gaussian_Vector;
}

vector<Mat> LaplacianPyramid_Func(Mat Image) { // Laplacian Pyramid를 만들어주는 Function

    vector<Mat> Laplacian_Vector; // 마지막에 return해줄 모든 영상을 저장할 벡터 매트릭스 선언

    for (int i = 0; i < 8; i++) { // 8 계층
        if (i < 7) {
            Mat Prev_Img = Image; // Filter 적용 이전 영상 백업

            Image = Sampling_Func(Image); // 영상 샘플링
            Image = GaussianFilter_Func(Image); // Gaussian filter 적용

            Mat Next_Img = Image; // filter를 적용한 다음 영상

            resize(Next_Img, Next_Img, Prev_Img.size()); // 다시 이전 영상 사이즈로 복원

            Laplacian_Vector.push_back(Prev_Img - Next_Img + 128); // Laplacian, DoG(Difference of Gaussian)
        }
        else {
            Laplacian_Vector.push_back(Image);
        }
    }
    return Laplacian_Vector;
}

Mat Make_Hybrid_Func(vector<Mat> Hybrid_Material) { // 재료들로 하이브리드 이미지를 만들어주는 함수

    Mat Hyb_I; // 최종적으로 만들어진 hybrid image를 담을 매트릭스 선언

    for (int i = 0; i < Hybrid_Material.size(); i++) {
        if (i == 0) { // 첫번째 Gaussian 이미지는 그대로 불러오기
            Hyb_I = Hybrid_Material[i];
        }
        else {
            resize(Hyb_I, Hyb_I, Hybrid_Material[i].size()); // 그 다음 재료의 사이즈로 조정
            Hyb_I = Hyb_I + Hybrid_Material[i] - 128; // 이후 더해주기 (over-flow 방지를 위해 128 빼주기)
        }
    }

    return Hyb_I;
}

int main() {

    // 1번

    Mat Img1, Img2; // 1,2번 영상 매트릭스 선언

    Img1 = imread("cat.jpg", 0); // 1번 영상 불러오기
    Img2 = imread("dog.jpg", 0); // 2번 영상 불러오기

    vector<Mat> Gaussian_Pyramid_Vector_1 = GaussianPyramid_Func(Img1);
    vector<Mat> Laplacian_Pyramid_Vector_1 = LaplacianPyramid_Func(Img1);

    vector<Mat> Gaussian_Pyramid_Vector_2 = GaussianPyramid_Func(Img2);
    vector<Mat> Laplacian_Pyramid_Vector_2 = LaplacianPyramid_Func(Img2);

    for (int i = 0; i < Gaussian_Pyramid_Vector_1.size(); i++) {
        imshow("Gaussian pyramid", Gaussian_Pyramid_Vector_1[i]);
        waitKey(0);
    }

    for (int i = 0; i < Laplacian_Pyramid_Vector_1.size(); i++) {
        imshow("Laplacian pyramid", Laplacian_Pyramid_Vector_1[i]);
        waitKey(0);
    }

    // 2번

    vector<Mat> Hybrid_Material; // 하이브리드 이미지의 재료들이 담길 벡터 매트릭스
    Mat Hyb_I; // 하이브리드 이미지 매트릭스 선언

    for (int i = 0; i < 8; i++) {
        if (i >= 0 && i <= N1) { // N1개의 Laplacian image
            Hybrid_Material.push_back(Laplacian_Pyramid_Vector_1[i]); 
        }
        else if (i >= (8 - N2)) { // N2개의 Laplacian image + Last Gaussian image
            Hybrid_Material.push_back(Laplacian_Pyramid_Vector_2[i]); 
        }
    }

    reverse(Hybrid_Material.begin(), Hybrid_Material.end());

    Hyb_I = Make_Hybrid_Func(Hybrid_Material);

    imwrite("Hybrid.jpg", Hyb_I);
    imshow("Hybrid", Hyb_I);
    waitKey(0);
}