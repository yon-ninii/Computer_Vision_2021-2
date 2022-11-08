#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;
using namespace cv;

const int max_disparity = 64;                    // 최대 시차, search range
const int win_col = 5;                           // window 열 개수
const int win_row = 5;                           // window 행 개수
const int win_size = win_col * win_row;          // window size
const int radius = (win_col - 1) / 2;            // window 반지름 (실제로는 반지름 - 0.5), for문 계산할 때 쓰일 parameter
const int normalize_const = 255 / max_disparity; // depth(disparity)의 최댓값을 255로 정규화 시켜주는 상수
const double gamma_C = 7;                        // weight 수식에 들어갈 γ(color)
const double gamma_D = (radius * 1.5);           // weight 수식에 들어갈 γ(distance)

double sim_weight_calc(double I_p, double I_q);                                   // similarity(ΔC) 를 구하는 함수
double prox_weight_calc(int p_x, int p_y, int q_x, int q_y);                      // proximity(ΔD) 를 구하는 함수
double adaptive_weight(double sim, double prox);                                  // similarity와 proximity를 이용해 W(p, q)를 구하는 함수
double SAD(double I_q, double I_qd);                                              // Sum of Absolute Difference
Mat disparity_calc(int max_col, int max_row, Mat left, Mat right);                // depth(disparity) map을 만들어주는 함수
int myKernelConv5x5(uchar *arr, Mat kernel, int x, int y, int width, int height); // 5x5 convolution function
Mat mySobelFilter(Mat srcImg);                                                    // sobel filter function

int main()
{
    Mat left_img = imread("adapt_image/aloe_left.png", 0);   // 좌 영상 입력
    Mat right_img = imread("adapt_image/aloe_right.png", 0); // 우 영상 입력

    Mat true_img = imread("true_image/aloe_true.png", 0); // ground truth 영상 입력

    Mat left_grad = mySobelFilter(left_img);   // gradient filter 적용
    Mat right_grad = mySobelFilter(right_img); // gradient filter 적용

    left_grad.convertTo(left_grad, CV_64FC1);   // double로 변환
    right_grad.convertTo(right_grad, CV_64FC1); // double로 변환
    true_img.convertTo(true_img, CV_8UC1);      // uchar로 변환

    int cols = left_img.cols; // 입력 영상의 열
    int rows = left_img.rows; // 입력 영상의 행

    Mat depth_map = Mat::zeros(rows, cols, CV_8UC1); // 결과를 담을 Matrix 초기화

    depth_map = disparity_calc(cols, rows, left_grad, right_grad); // 결과물 생성

    Mat error_map = Mat::zeros(rows, cols, CV_8UC1); // 에러를 표시해줄 Matrix 초기화

    error_map = true_img - depth_map; // ground truth - depth 로 에러 맵 생성

    imwrite("sobel_result/aloe_result_sobel.png", depth_map); // 결과 파일 생성
    imshow("result", depth_map);                              // 결과 출력

    waitKey(0);
    destroyAllWindows();

    imwrite("sobel_result/aloe_error_sobel.png", error_map); // 에러 맵 파일 생성
    imshow("error", error_map);                              // 에러 맵 출력

    waitKey(0);
    destroyAllWindows();

    return 0;
}

double sim_weight_calc(double I_p, double I_q)
{
    return fabs(I_p - I_q); // floating absolute 함수 사용
}

double prox_weight_calc(int p_x, int p_y, int q_x, int q_y)
{
    return hypot(p_x - q_x, p_y - q_y); // 두 점 사이의 거리를 구해주는 함수 hypot 사용
}

double adaptive_weight(double sim, double prox)
{
    return exp(-((sim / gamma_C) + (prox / gamma_D))); // w(p, q) 함수 구현
}

double SAD(double I_q, double I_qd)
{
    return fabs(I_q - I_qd); // e(q, qd) 함수 구현
}

double SSD(double C_q, double C_qd)
{
    return pow(C_q - C_qd, 2); // e(q, qd) 함수 구현
}

Mat disparity_calc(int max_col, int max_row, Mat left, Mat right) // 최종 결과물을 생성해줄 함수
{
    Mat disparity_map = Mat::zeros(max_row, max_col, CV_8UC1); // 결과물을 담을 matrix 선언 및 초기화

    for (int row = 0; row < max_row; row++) // 총 행의 개수만큼 반복
    {
        for (int column = 0; column < max_col; column++) // 총 열의 개수만큼 반복
        {
            int disparity = 0;     // 시차 값 담아줄 int형 변수
            double min_cost = 0.0; // 최소 비용을 담아줄 double형 변수

            for (int i = 0; i < max_disparity; i++) // 최대 시차(탐색 범위)만큼 반복 (왼쪽으로 sliding window 시킨다)
            {
                double cost_num = 0.0;   // 비용 함수 분자
                double cost_denom = 0.0; // 비용 함수 분모

                int min_x = max(0, column - radius - i); // 탐색 범위 or window size로 인해 영상 바깥의 인덱스까지 침범하는 것을 방지
                int min_y = max(0, row - radius);        //window 만큼 내부 for문을 돌리기 위한 예외처리 작업
                int max_x = min(column + radius - i, max_col - 1);
                int max_y = min(row + radius, max_row - 1);

                for (int y = min_y; y <= max_y; y++) // window 내부 cost 계산 for문
                {
                    for (int x = min_x; x <= max_x; x++) // window 내부 cost 계산 for문
                    {

                        double proximity_l = 0.0;
                        double similarity_l = 0.0;
                        double proximity_r = 0.0;
                        double similarity_r = 0.0;
                        double error = 0.0;
                        double left_weight = 0.0;
                        double right_weight = 0.0;

                        similarity_l = sim_weight_calc(left.at<double>(row, column), left.at<double>(y, x + i)); // 좌 영상 similarity 계산
                        proximity_l = prox_weight_calc(column, row, x + i, y);                                   // 좌 영상 proximity 계산

                        similarity_r = sim_weight_calc(right.at<double>(row, column - i), right.at<double>(y, x)); // 우 영상 similarity 계산
                        proximity_r = prox_weight_calc(column - i, row, x, y);                                     // 우 영상 proximity 계산

                        error = SSD(left.at<double>(y, x + i), right.at<double>(y, x)); // e(q, qd)로 SAD error 계산

                        left_weight = adaptive_weight(similarity_l, proximity_l);  // 좌 영상 가중치 w(p, q) 계산
                        right_weight = adaptive_weight(similarity_r, proximity_r); // 우 영상 가중치 w(pd, qd) 계산

                        cost_num += left_weight * right_weight * error; // 비용 함수 분자 계산
                        cost_denom += left_weight * right_weight;       // 비용 함수 분모 계산
                    }
                }
                double cost = (cost_num / cost_denom); // 비용 함수 분자/분모 꼴로 만들어주기

                if (i == 0)
                    min_cost = cost; // 최소 비용 초기화
                if (cost < min_cost)
                {
                    min_cost = cost; // 최소 비용 업데이트하면서 시차 찾기
                    disparity = i;
                }
            }
            disparity_map.at<uchar>(row, column) = (uchar)(normalize_const * disparity); // 찾은 시차 결과 matrix에 삽입
        }
    }
    return disparity_map; // 결과값 리턴
}

int myKernelConv5x5(uchar *arr, int kernel[][5], int x, int y, int width, int height) // 5x5 convolution function
{
    int sum = 0;
    int sumKernel = 0;

    for (int j = -2; j <= 2; j++) // 5x5만큼 for문
    {
        for (int i = -2; i <= 2; i++)
        {
            if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
            {
                sum += arr[(y + j) * width + (x + i)] * kernel[i + 2][j + 2]; // 컨볼루션 적용
            }
        }
    }
    return sum;
}

Mat mySobelFilter(Mat srcImg) // sobel filter function
{
    int kernelX[5][5] = {-2, -1, 0, 1, 2, // x방향 sobel kernel
                         -2, -1, 0, 1, 2,
                         -4, -2, 0, 2, 4,
                         -2, -1, 0, 1, 2,
                         -2, -1, 0, 1, 2};

    int kernelY[5][5] = {-2, -2, -4, -2, -2, // y방향 sobel kernel
                         -1, -1, -2, -1, -1,
                         0, 0, 0, 0, 0,
                         1, 1, 2, 1, 1,
                         2, 2, 4, 2, 2};

    Mat dstImg(srcImg.size(), CV_8UC1);
    uchar *srcData = srcImg.data;
    uchar *dstData = dstImg.data;
    int width = srcImg.cols;
    int height = srcImg.rows;

    for (int y = 0; y < height; y++) // 전체 크기만큼 컨볼루션 진행 + hypot 함수로 바로 magnitude of gradient 계산
    {
        for (int x = 0; x < width; x++)
        {
            dstData[y * width + x] = hypot(myKernelConv5x5(srcData, kernelX, x, y, width, height), myKernelConv5x5(srcData, kernelY, x, y, width, height));
        }
    }
    return dstImg;
}