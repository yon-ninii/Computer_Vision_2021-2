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

const int max_disparity = 48;					 // 최대 시차, search range
const int win_col = 17;							 // window 열 개수
const int win_row = 17;							 // window 행 개수
const int win_size = win_col * win_row;			 // window size
const int radius = (win_col - 1) / 2;			 // window 반지름 (실제로는 반지름 - 0.5), for문 계산할 때 쓰일 parameter
const int normalize_const = 255 / max_disparity; // depth(disparity)의 최댓값을 255로 정규화 시켜주는 상수
const double gamma_C = 7;						 // weight 수식에 들어갈 γ(color)
const double gamma_D = (radius * 1.5);			 // weight 수식에 들어갈 γ(distance)

double sim_weight_calc(double I_p, double I_q);					   // similarity(ΔC) 를 구하는 함수
double prox_weight_calc(int p_x, int p_y, int q_x, int q_y);	   // proximity(ΔD) 를 구하는 함수
double adaptive_weight(double sim, double prox);				   // similarity와 proximity를 이용해 W(p, q)를 구하는 함수
double SAD(double I_q, double I_qd);							   // Sum of Absolute Difference
Mat disparity_calc(int max_col, int max_row, Mat left, Mat right); // depth(disparity) map을 만들어주는 함수

int main()
{
	Mat left_img = imread("adapt_image/teddy_left.png", 0);	  // 좌 영상 입력
	Mat right_img = imread("adapt_image/teddy_right.png", 0); // 우 영상 입력

	Mat true_img = imread("true_image/teddy_true.png", 0); // ground truth 영상 입력

	left_img.convertTo(left_img, CV_64FC1);	  // double로 변환
	right_img.convertTo(right_img, CV_64FC1); // double로 변환
	true_img.convertTo(true_img, CV_8UC1);	  // uchar로 변환

	int cols = left_img.cols; // 입력 영상의 열
	int rows = left_img.rows; // 입력 영상의 행

	Mat depth_map = Mat::zeros(rows, cols, CV_8UC1); // 결과를 담을 Matrix 초기화

	depth_map = disparity_calc(cols, rows, left_img, right_img); // 결과물 생성

	Mat error_map = Mat::zeros(rows, cols, CV_8UC1); // 에러를 표시해줄 Matrix 초기화

	error_map = true_img - depth_map; // ground truth - depth 로 에러 맵 생성

	imwrite("adapt_result/aloe_result.png", depth_map); // 결과 파일 생성
	imshow("result", depth_map);						// 결과 출력

	waitKey(0);
	destroyAllWindows();

	imwrite("adapt_result/aloe_error.png", error_map); // 에러 맵 파일 생성
	imshow("error", error_map);						   // 에러 맵 출력

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

Mat disparity_calc(int max_col, int max_row, Mat left, Mat right) // 최종 결과물을 생성해줄 함수
{
	Mat disparity_map = Mat::zeros(max_row, max_col, CV_8UC1); // 결과물을 담을 matrix 선언 및 초기화

	for (int row = 0; row < max_row; row++) // 총 행의 개수만큼 반복
	{
		for (int column = 0; column < max_col; column++) // 총 열의 개수만큼 반복
		{
			int disparity = 0;	   // 시차 값 담아줄 int형 변수
			double min_cost = 0.0; // 최소 비용을 담아줄 double형 변수

			for (int i = 0; i < max_disparity; i++) // 최대 시차(탐색 범위)만큼 반복 (왼쪽으로 sliding window 시킨다)
			{
				double cost_num = 0.0;	 // 비용 함수 분자
				double cost_denom = 0.0; // 비용 함수 분모

				int min_x = max(0, column - radius - i); // 탐색 범위 or window size로 인해 영상 바깥의 인덱스까지 침범하는 것을 방지
				int min_y = max(0, row - radius);		 //window 만큼 내부 for문을 돌리기 위한 예외처리 작업
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
						proximity_l = prox_weight_calc(column, row, x + i, y);									 // 좌 영상 proximity 계산

						similarity_r = sim_weight_calc(right.at<double>(row, column - i), right.at<double>(y, x)); // 우 영상 similarity 계산
						proximity_r = prox_weight_calc(column - i, row, x, y);									   // 우 영상 proximity 계산

						error = SAD(left.at<double>(y, x + i), right.at<double>(y, x)); // e(q, qd)로 SAD error 계산

						left_weight = adaptive_weight(similarity_l, proximity_l);  // 좌 영상 가중치 w(p, q) 계산
						right_weight = adaptive_weight(similarity_r, proximity_r); // 우 영상 가중치 w(pd, qd) 계산

						cost_num += left_weight * right_weight * error; // 비용 함수 분자 계산
						cost_denom += left_weight * right_weight;		// 비용 함수 분모 계산
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