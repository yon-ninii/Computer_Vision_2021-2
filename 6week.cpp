#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp> // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include <opencv2/highgui/highgui.hpp> // GUI 와 관련된 요소를 포함하는 헤더 imshow 등
#include <opencv2/imgproc/imgproc.hpp> // 각종 이미지 처리 함수를 포함하는 헤더
#include <ctime>


using namespace std;
using namespace cv;
double gaussian(float x, double sigma);
double gaussian2D(float c, float r, double sigma);
float distance(int x, int y, int i, int j);
void myGaussian(const Mat& src_img, Mat& dst_img, Size size);
void mykernelConv(const Mat& src_img, Mat& dst_img, const Mat& kn);
int selectionSort_median(float arr[], int size);
void median(float table[], uchar* src_data, int c, int r, int hg, int rad_w, int rad_h);
void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size);
void doMedianEx();
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s);
void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s);
void doBilateralEx();
void doCannyEx();

int main() {
	doMedianEx();
	doCannyEx();
	doBilateralEx();
}
double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2)))
		/ (2 * CV_PI * pow(sigma, 2));
}
float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}
double gaussian2D(float c, float r, double sigma) {
	return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2)))
		/ (2 * CV_PI * pow(sigma, 2));
}

void myGaussian(const Mat& src_img, Mat& dst_img, Size size) {
	//<커널 생성>
	Mat kn = Mat::zeros(size, CV_32FC1);
	double sigma = 1.0;
	float* kn_data = (float*)kn.data;
	for (int c = 0; c < kn.cols; c++) {
		for (int r = 0; r < kn.rows; r++) {
			kn_data[r * kn.cols + c] =
				(float)gaussian2D((float)(c - kn.cols / 2),
					(float)(r - kn.rows / 2), sigma); //cv_32f err type casting 
		}
	}
	// <커널 컨볼루션 수행>
	mykernelConv(src_img, dst_img, kn);
}
void mykernelConv(const Mat& src_img, Mat& dst_img, const Mat& kn) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn.cols; int khg = kn.rows;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	float* kn_data = (float*)kn.data;
	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float wei, tmp, sum;

	// <픽셀 인덱싱 (가장자리 제외)>
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			sum = 0.f;
			// <커널 인덱싱>
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
					tmp += wei * (float)src_data[(r + kr) * wd + (c + kc)];
					sum += wei;
				}
			}
			if (sum != 0.f) tmp = abs(tmp) / sum; //정규화 및 overflow 방지
			else tmp = abs(tmp);

			if (tmp > 255.f)tmp = 255.f; //overflow 방지

			dst_data[r * wd + c] = (uchar)tmp;
		}
	}
}

// median functions
int selectionSort_median(float arr[],int size) {
	for (int i = 0; i < size-1; i++) {
		int  min = i;
		for (int j = i + 1; j < size; j++) {

			if (arr[min] > arr[j]) {
				int temp = arr[min];
				arr[min] = arr[j];
				arr[j] = temp;
			}
		}
	}
	return arr[size/2];
}
void median(float table[], uchar* src_data, int c, int r, int hg, int rad_w,int rad_h){
	int k = 0;
	for (int i = -rad_w; i <= rad_w; i++) {
		for (int j = -rad_h; j <= rad_h; j++, k++) {
			table[k] = src_data[((c + i) * hg) + (r + j)];
		}
	}
}

void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn_size.width; int khg = kn_size.height;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float* table = new float[kwd * khg](); // 커널 테이블 동적할당
	float tmp;

	int size = kwd * khg;
	// <픽셀 인덱싱 (가장자리 제외)>
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			median(table, src_data, c, r, hg,rad_w,rad_h);
			dst_data[c * hg + r] = selectionSort_median(table, size);

		}
	}
	delete table;
}

void doMedianEx() {
	cout << "--- doMedianEX() --- " << endl;

	// <Input>
	Mat src_img = imread("salt_pepper2.png", 0);
	if (!src_img.data) printf("No image data \n");

	// <Median 필터링 수행>
	Mat dst_img;
	myMedian(src_img, dst_img, Size(5, 5));

	// <Output>
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doMedianEx()", result_img);
	waitKey(0);
}
// bilateral functions
void bilateral(const Mat& src_img, Mat& dst_img,
	int c, int r, int diameter, double sig_r, double sig_s) {
	int radius = diameter / 2;

	double gr, gs, wei;
	double tmp = 0.;
	double sum = 0.;

	// <커널 인덱싱>
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr)
				- (float)src_img.at<uchar>(c, r), sig_r);
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; //정규화
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);
	
	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;
	
	//<픽셀 인덱싱(가장자리 제외)>
	int cnt = 0;
	int c ,r;
	
	for (c = radius + 1; c < hg - radius; c++) {
		for (r = radius + 1; r < wh - radius; r++) {
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
			cnt++;
			//화소별 Bilateral 계산 수행
		}

	}
	
	guide_img.convertTo(dst_img, CV_8UC1); // Mat type 변환
}

void doBilateralEx() {
	cout << "--- doBilateralEX() --- " << endl;

	// <Input>
	Mat src_img = imread("rock.png", 0);
	if (!src_img.data) printf("No image data \n");
	// <Bilateral 필터링 수행>
	
	Mat dst_img1;	Mat dst_img2;	Mat dst_img3;	Mat dst_img4;	Mat dst_img5;
	Mat dst_img6;	Mat dst_img7;	Mat dst_img8;	Mat dst_img9;
	myBilateral(src_img, dst_img1, 40, 30, 1); 
	myBilateral(src_img, dst_img2, 40, 80, 1);
	myBilateral(src_img, dst_img3, 40, SIZE_MAX, 1);
	myBilateral(src_img, dst_img4, 40, 30, 5);
	myBilateral(src_img, dst_img5, 40, 80, 5);
	myBilateral(src_img, dst_img6, 40, SIZE_MAX, 5);
	myBilateral(src_img, dst_img7, 40, 30, 30);
	myBilateral(src_img, dst_img8, 40, 80, 30);
	myBilateral(src_img, dst_img9, 40, SIZE_MAX, 30);

	// <Output>
	Mat result_img1;
	Mat result_img2;
	Mat result_img3;
	hconcat(dst_img1, dst_img2, result_img1);
	hconcat(result_img1, dst_img3, result_img1);

	hconcat(dst_img4, dst_img5, result_img2);
	hconcat(result_img2, dst_img6, result_img2);

	hconcat(dst_img7, dst_img8, result_img3);
	hconcat(result_img3, dst_img9, result_img3);

	vconcat(result_img1, result_img2, result_img1);
	vconcat(result_img1, result_img3, result_img1);
	
	imshow("doBilateralEx()", result_img1);
	waitKey(0);
}

void doCannyEx() {
	cout << "--- doCannyEx() -- \n" << endl;
	clock_t start, end;
	
	// <입력>
	Mat src_img = imread("rock.png", 0);
	if (!src_img.data) printf("No image data \n");

	Mat dst_img;
	//<Canny edge 탐색 수행>

	start = clock();
	Canny(src_img, dst_img, 200, 240);
	end = clock();
	cout << "걸린 시간: " << end - start << "ms" << endl;
	cout << endl;
	

	// <출력>
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("rock", result_img);
	waitKey(0);
}

