#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include <time.h>
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

void Gaussian_Elimination_func(float* input, int n){ // 행렬로 이루어진 선형 방정식을 풀어주는 Gaussian Elimination 함수

    float* A = input;
    int i = 0;
    int j = 0;
    // m = rows, n = cols
    int m = n - 1; // 8 과 9 이므로
    while (i < m && j < n)
    {
        // column j를 기준으로 잡아서 row i부터 시작
        int maxi = i;
        for (int k = i + 1; k < m; k++)
        {
            if (fabs(A[k * n + j]) > fabs(A[maxi * n + j]))
            {
                maxi = k;
            }
        }
        if (A[maxi * n + j] != 0)
        {
            // i와 maxi의 순서를 바꾼다. (위부터 큰 순서대로 sort 돼야 하기 때문)
            if (i != maxi)
                for (int k = 0; k < n; k++)
                {
                    float aux = A[i * n + k];
                    A[i * n + k] = A[maxi * n + k];
                    A[maxi * n + k] = aux;
                }
            // A[i,j] 바꾸기 전 A[maxi,j]의 값을 가지고 있는 상태
            // 행 i의 원소를 A[i,j]로 나눠준다
            float A_ij = A[i * n + j];
            for (int k = 0; k < n; k++)
            {
                A[i * n + k] /= A_ij;
            }
            // A[i,j] = 1이 됐을 것이다.
            for (int u = i + 1; u < m; u++)
            {
                // A[u,j]와 행 i를 곱한 값에서 행 u를 빼준다.
                float A_uj = A[u * n + j];
                for (int k = 0; k < n; k++)
                {
                    A[u * n + k] -= A_uj * A[i * n + k];
                }
                // A[u,j] = 0 이다
            }
            i++;
        }
        j++;
    }

    // 다시 대입해준다
    for (int i = m - 2; i >= 0; i--)
    {
        for (int j = i + 1; j < n - 1; j++)
        {
            A[i * n + m] -= A[i * n + j] * A[j * n + m];
        }
    }
}

void findHomography_func(vector<Point2f> src, vector<Point2f> dst, Mat & result){ // Homography를 찾아주는 함수

    float P[8][9] =                                                              // 선형 방정식의 앞쪽 행렬이다.
    {
        {-src[0].x, -src[0].y, -1,   0,   0,  0, src[0].x * dst[0].x, src[0].y * dst[0].x, -dst[0].x }, 
        {  0,   0,  0, -src[0].x, -src[0].y, -1, src[0].x * dst[0].y, src[0].y * dst[0].y, -dst[0].y },

        {-src[1].x, -src[1].y, -1,   0,   0,  0, src[1].x * dst[1].x, src[1].y * dst[1].x, -dst[1].x },
        {  0,   0,  0, -src[1].x, -src[1].y, -1, src[1].x * dst[1].y, src[1].y * dst[1].y, -dst[1].y }, 

        {-src[2].x, -src[2].y, -1,   0,   0,  0, src[2].x * dst[2].x, src[2].y * dst[2].x, -dst[2].x },
        {  0,   0,  0, -src[2].x, -src[2].y, -1, src[2].x * dst[2].y, src[2].y * dst[2].y, -dst[2].y },

        {-src[3].x, -src[3].y, -1,   0,   0,  0, src[3].x * dst[3].x, src[3].y * dst[3].x, -dst[3].x },
        {  0,   0,  0, -src[3].x, -src[3].y, -1, src[3].x * dst[3].y, src[3].y * dst[3].y, -dst[3].y },
    };

    Gaussian_Elimination_func(&P[0][0], 9); // 해당 선형 방정식을 가우시안 소거법을 사용해서 풀어준다.

    float aux_H[3][3] = { // 소거법을 활용해 풀어준 결과 행렬
        { P[0][8], P[1][8], P[2][8] },
        { P[3][8], P[4][8], P[5][8] },	
        { P[6][8], P[7][8], 1 }
    };	
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.at<float>(i, j) = aux_H[i][j]; // 결과 Mat에 대입해줘서 출력
        }
    }
}


Mat RANSAC_func(vector<Point2f> src, vector<Point2f> dst){ // RANSAC을 수행해주는 함수

        int N = 2000;	// 반복 횟수
        float T = 3;   // threshold
        int size = src.size(); // 입력 keypoint들의 개수

        int max_cnt = 0; // 가장 많은 inlier의 개수를 담아줄 변수

        Mat Best_Homo = Mat::zeros(3, 3, CV_32FC1); // 가장 많은 inlier를 뽑아낸 호모그래피를 저장해줄 Mat 초기화

        for (int i = 0; i < N; i++) // N번 반복해주는 for문
        {
            int k[4] = { -1, };
            k[0] = floor((rand() % size)); // 4개씩 총 keypoint의 개수 범위 내에서 만큼 난수를 생성해주는 난수 생성기
            do
            {
                k[1] = floor((rand() % size));
            } while (k[1] == k[0] || k[1] < 0);

            do
            {
                k[2] = floor((rand() % size));
            } while (k[2] == k[0] || k[2] == k[1] || k[2] < 0);

            do
            {
                k[3] = floor((rand() % size));
            } while (k[3] == k[0] || k[3] == k[1] || k[3] == k[2] || k[3] < 0);

            printf("random sample : %d %d %d %d\n", k[0], k[1], k[2], k[3]); // 몇번째 특징점들이 뽑혔는지 출력

            vector<Point2f> src_sample; // 뽑힌 번호들의 특징점(x)들을 담아둘 컨테이너
            vector<Point2f> dst_sample; // 뽑힌 번호들의 특징점(x')들을 담아둘 컨테이너
            
            src_sample.push_back(src[k[0]]);
            src_sample.push_back(src[k[1]]);
            src_sample.push_back(src[k[2]]);
            src_sample.push_back(src[k[3]]);

            dst_sample.push_back(dst[k[0]]);
            dst_sample.push_back(dst[k[1]]);
            dst_sample.push_back(dst[k[2]]);
            dst_sample.push_back(dst[k[3]]);

            Mat Homo_sample = Mat::zeros(3, 3, CV_32FC1); // 뽑힌 특징점들로 만든 호모그래피를 담을 3 x 3 매트릭스 초기화 및 선언

            findHomography_func(src_sample, dst_sample, Homo_sample); // find homography 함수 선언
                                    
            Mat Distance = Mat::zeros(size, 1, CV_32FC1); // 호모그래피의 거리를 계산한 값들을 담을 (size) x 1 매트릭스

            for (int l = 0; l < size; l++) { // 특징점 개수만큼 반복
                Mat A_calc = Mat::zeros(3, 1, CV_32FC1); // 특징점을 Mat으로 옮기는 것과 동시에 homogeneous coordinate로 변환해서 담을 3 x 1 매트릭스
                Point2f A_calc_Homo; // H x X 를 계산하여 다시 cartesian coordinate의 변환해서 담아둘 변수
                                
                A_calc.at<float>(0, 0) = src[l].x;
                A_calc.at<float>(1, 0) = src[l].y;
                A_calc.at<float>(2, 0) = 1.0f; // homogeneous coordinate로 변환

                Mat H_mul_X = (3, 1, CV_32FC1, Homo_sample * A_calc); // H x X 계산

                A_calc_Homo.x = H_mul_X.at<float>(0, 0) / H_mul_X.at<float>(2, 0); // (wx, wy, w)에서 wx / w를 계산해서 x 값 삽입
                A_calc_Homo.y = H_mul_X.at<float>(1, 0) / H_mul_X.at<float>(2, 0); // (wx, wy, w)에서 wy / w를 계산해서 y 값 삽입

                Distance.at<float>(l, 0) = norm(dst[l] - A_calc_Homo); // norm 함수를 이용해서 거리 계산
            }

            int cnt = 0; // inlier의 개수를 저장해줄 변수

            for (int j = 0; j < size; j++)
            {
                float data = Distance.at<float>(j, 0); // data에 거리 옮기기
                if (data < T) // threshold 와 비교
                {
                    cnt++;
                }
            }

            if (cnt > max_cnt) // 가장 많은 inlier의 개수를 가진 호모그래피 저장
            {
                Best_Homo = Homo_sample;
                max_cnt = cnt;
            }
        }

        cout << "Max inliers : " << max_cnt << endl;
        cout << "Best Homography : " << Best_Homo << endl << endl;

        return Best_Homo;
}

Mat makePanorama_func(Mat Left_Img, Mat Right_Img, int Distance_Thresh, int Match_Min) { // 입력된 두 이미지를 병합해주는 함수

    Mat Left_Img_Gray, Right_Img_Gray; // Grayscale로 변환한 이미지를 담아둘 매트릭스

    cvtColor(Left_Img, Left_Img_Gray, CV_BGR2GRAY); // Grayscale로 변환
    cvtColor(Right_Img, Right_Img_Gray, CV_BGR2GRAY); // Grayscale로 변환

    // --------------------SIFT 특징 검출기------------------------------

    Ptr<SiftFeatureDetector> Detector = SIFT::create(300); 
    vector<KeyPoint> kpts_left, kpts_right;
    Detector->detect(Left_Img_Gray, kpts_left);
    Detector->detect(Right_Img_Gray, kpts_right);

    Ptr<SiftDescriptorExtractor> Extractor = SIFT::create(100, 4, 3, false, true);
    Mat img_des_left, img_des_right;
    Extractor->compute(Left_Img_Gray, kpts_left, img_des_left);
    Extractor->compute(Right_Img_Gray, kpts_right, img_des_right);

    BFMatcher matcher(NORM_L2); // Brute Force Matcher -- L2 norm 사용
    vector<DMatch> matches;
    matcher.match(img_des_left, img_des_right, matches);

    double Distance_Max = matches[0].distance; // Match들의 거리를 통해 정제
    double Distance_Min = matches[0].distance;
    double dist;
    for (int i = 0; i < img_des_left.rows; i++) {
        dist = matches[i].distance;
        if (dist < Distance_Min) Distance_Min = dist;
        if (dist > Distance_Max) Distance_Max = dist;
    }
    printf("max_dist : %f \n", Distance_Max);
    printf("min_dist : %f \n", Distance_Min);

    vector<DMatch> Matched_Fin;
    do {
        vector<DMatch> Matched_Good;
        for (int i = 0; i < img_des_left.rows; i++) {
            if (matches[i].distance < Distance_Thresh * Distance_Min)
                Matched_Good.push_back(matches[i]);
        }
        Matched_Fin = Matched_Good;
        Distance_Thresh -= 1;
    } while (Distance_Thresh != 2 && Matched_Fin.size() > Match_Min);

    vector<Point2f> left, right; // Match가 끝난 페어들의 좌표를 각각 left와 right 벡터에 저장
    for (int i = 0; i < Matched_Fin.size(); i++) {
        left.push_back(kpts_left[Matched_Fin[i].queryIdx].pt);
        right.push_back(kpts_right[Matched_Fin[i].trainIdx].pt);
    }
    cout <<  "Numnber of Keypoints : " << right.size() << ", " << left.size() << endl; // 특징점 총 개수 출력

    //Mat mat_homo = findHomography(left, right, RANSAC); // built-in과 나의 함수 비교

    Mat mat_homo = RANSAC_func(left, right); // Homography를 만들어줌과 동시에 RANSAC algorithm을 이용해 가장 좋은 Homography를 찾아주는 함수

    // cout << "Built-in Homography : " <<mat_homo_1 << endl << endl; // built-in과 나의 함수 비교

    Mat Result_Img;
    warpPerspective(Left_Img, Result_Img, mat_homo, // Inverse Warping을 통해 projection 실행
        Size(Left_Img.cols * 3, Left_Img.rows * 1.2), INTER_CUBIC);

    Mat Pano_Img;
    Pano_Img = Result_Img.clone();
    Mat roi(Pano_Img, Rect(0, 0, Right_Img.cols, Right_Img.rows)); // 이미지 합성
    Right_Img.copyTo(roi);

    int cut_x = 0, cut_y = 0; 
    for (int y = 0; y < Pano_Img.rows; y++) { // Black Cut - 이미지에서 검은색 부분을 잘라주는 코드
        for (int x = 0; x < Pano_Img.cols; x++) {
            if (Pano_Img.at<Vec3b>(y, x)[0] == 0 &&
                Pano_Img.at<Vec3b>(y, x)[1] == 0 &&
                Pano_Img.at<Vec3b>(y, x)[2] == 0) {
                continue;
            }
            if (cut_x < x)cut_x = x;
            if (cut_y < y)cut_y = y;
        }
    }
    Mat Pano_Img_Fin;
    Pano_Img_Fin = Pano_Img(Range(0, cut_y), Range(0, cut_x));

    return Pano_Img_Fin;
}

void Panorama_func() { // 파노라마를 전체적으로 진행해주는 함수

    Mat Image_left = imread("home1.jpg", IMREAD_COLOR); // 세장의 사진 불러오기
    Mat Image_center = imread("home2.jpg", IMREAD_COLOR);
    Mat Image_right = imread("home3.jpg", IMREAD_COLOR);

    resize(Image_left, Image_left, Size(512, 512), INTER_AREA); // 모두 같은 사이즈로 다운 샘플링
    resize(Image_center, Image_center, Size(512, 512), INTER_AREA);
    resize(Image_right, Image_right, Size(512, 512), INTER_AREA);

    if (Image_left.empty() || Image_center.empty() || Image_right.empty()) exit(-1); // 위 내용이 제대로 진행 안되었으면 종료

    Mat result; // 결과를 담아줄 매트릭스

    flip(Image_left, Image_left, 1); // 왼쪽 사진 뒤집기
    flip(Image_center, Image_center, 1); // 가운데 사진 뒤집기

    result = makePanorama_func(Image_left, Image_center, 4, 100); // 왼쪽과 가운데 먼저 병합

    flip(result, result, 1); // 왼쪽 + 가운데 다시 뒤집기
    imshow("ex_panorama_result1", result);
    imwrite("ex_panorama_result1.png", result);

    waitKey();

    result = makePanorama_func(Image_right, result, 2, 60); // (왼쪽 + 가운데) 와 오른쪽 병합

    imshow("ex_panorama_result2", result);
    imwrite("ex_panorama_result2.png", result);

    waitKey();

    destroyAllWindows();
}

int main() {
    Panorama_func();
}