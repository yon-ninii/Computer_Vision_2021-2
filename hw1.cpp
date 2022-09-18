#include <iostream>
#include <vector>
#include <algorithm>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp> // opencv���� �ʿ��� ��� ���� �ҷ�����

using namespace std;
using namespace cv; // cv�� std namespace ����

int N1 = 2; // �������� N1
int N2 = 5; // �������� N2

int conv3x3_Func(uchar *arr, int a, int b, int width, int height)
{ // 3x3 ������� �Լ�
    //a, b : rows, columns
    int kernel[3][3] = {1, 2, 1,
                        2, 4, 2,
                        1, 2, 1}; // 3x3 ����ũ �� �־��ֱ�

    int sum = 0;
    int sumKernel = 0;

    for (int j = -1; j < 2; j++)
    {
        for (int i = -1; i < 2; i++)
        { // 2�� for��
            if ((b + j) >= 0 && (b + j) < height && (a + i) >= 0 && (a + i) < width)
            {                                                                 // i,j��° �ȼ��� ������ �� ũ�� ���ο� ���� ��
                sum += arr[(b + j) * width + (a + i)] * kernel[i + 1][j + 1]; // ������� ���� ����
                sumKernel += kernel[i + 1][j + 1];
            }
        }
    }

    if (sumKernel == 0)
    {
        return sum;
    }
    else
    {
        return (sum / sumKernel);
    }
}

Mat GaussianFilter_Func(Mat Image)
{ // gaussian filter�� �����ϴ� �Լ�

    int width = Image.cols;
    int height = Image.rows;

    Mat Final_Img(Image.size(), CV_8UC1);

    uchar *Prev_Data = Image.data;
    uchar *Final_Data = Final_Img.data;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {                                                                             // 2�� for��
            Final_Data[j * width + i] = conv3x3_Func(Prev_Data, i, j, width, height); // 3x3 ������� �������ֱ�
        }
    }

    return Final_Img;
}

Mat Sampling_Func(Mat Image)
{ // ���ø� �Լ�

    int width = Image.cols / 2;  // �� �������� �ٿ��ֱ�
    int height = Image.rows / 2; // ���� �������� �ٿ��ֱ�

    Mat dstImg(height, width, CV_8UC1);

    uchar *srcData = Image.data;
    uchar *dstData = dstImg.data;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            dstData[j * width + i] = srcData[(j * 2) * (Image.cols) + (i * 2)]; // �� ��ĭ�� ����� �־��ֱ�
        }
    }

    return dstImg;
}

vector<Mat> GaussianPyramid_Func(Mat Image)
{ // Gaussian Pyramid�� ������ִ� Function

    vector<Mat> Gaussian_Vector; // �������� return���� ��� ������ ������ ���� ��Ʈ���� ����

    Gaussian_Vector.push_back(Image);

    for (int i = 0; i < 8; i++)
    {                                       // 8 ����
        Image = Sampling_Func(Image);       // ���� ���ø�
        Image = GaussianFilter_Func(Image); // Gaussian filter ����

        Gaussian_Vector.push_back(Image);
    }
    return Gaussian_Vector;
}

vector<Mat> LaplacianPyramid_Func(Mat Image)
{ // Laplacian Pyramid�� ������ִ� Function

    vector<Mat> Laplacian_Vector; // �������� return���� ��� ������ ������ ���� ��Ʈ���� ����

    for (int i = 0; i < 8; i++)
    { // 8 ����
        if (i < 7)
        {
            Mat Prev_Img = Image; // Filter ���� ���� ���� ���

            Image = Sampling_Func(Image);       // ���� ���ø�
            Image = GaussianFilter_Func(Image); // Gaussian filter ����

            Mat Next_Img = Image; // filter�� ������ ���� ����

            resize(Next_Img, Next_Img, Prev_Img.size()); // �ٽ� ���� ���� ������� ����

            Laplacian_Vector.push_back(Prev_Img - Next_Img + 128); // Laplacian, DoG(Difference of Gaussian)
        }
        else
        {
            Laplacian_Vector.push_back(Image);
        }
    }
    return Laplacian_Vector;
}

Mat Make_Hybrid_Func(vector<Mat> Hybrid_Material)
{ // ����� ���̺긮�� �̹����� ������ִ� �Լ�

    Mat Hyb_I; // ���������� ������� hybrid image�� ���� ��Ʈ���� ����

    for (int i = 0; i < Hybrid_Material.size(); i++)
    {
        if (i == 0)
        { // ù��° Gaussian �̹����� �״�� �ҷ�����
            Hyb_I = Hybrid_Material[i];
        }
        else
        {
            resize(Hyb_I, Hyb_I, Hybrid_Material[i].size()); // �� ���� ����� ������� ����
            Hyb_I = Hyb_I + Hybrid_Material[i] - 128;        // ���� �����ֱ� (over-flow ������ ���� 128 ���ֱ�)
        }
    }

    return Hyb_I;
}

int main()
{

    // 1��

    Mat Img1, Img2; // 1,2�� ���� ��Ʈ���� ����

    Img1 = imread("cat.jpg", 0); // 1�� ���� �ҷ�����
    Img2 = imread("dog.jpg", 0); // 2�� ���� �ҷ�����

    vector<Mat> Gaussian_Pyramid_Vector_1 = GaussianPyramid_Func(Img1);
    vector<Mat> Laplacian_Pyramid_Vector_1 = LaplacianPyramid_Func(Img1);

    vector<Mat> Gaussian_Pyramid_Vector_2 = GaussianPyramid_Func(Img2);
    vector<Mat> Laplacian_Pyramid_Vector_2 = LaplacianPyramid_Func(Img2);

    for (int i = 0; i < Gaussian_Pyramid_Vector_1.size(); i++)
    {
        imshow("Gaussian pyramid", Gaussian_Pyramid_Vector_1[i]);
        waitKey(0);
    }

    for (int i = 0; i < Laplacian_Pyramid_Vector_1.size(); i++)
    {
        imshow("Laplacian pyramid", Laplacian_Pyramid_Vector_1[i]);
        waitKey(0);
    }

    // 2��

    vector<Mat> Hybrid_Material; // ���̺긮�� �̹����� ������ ��� ���� ��Ʈ����
    Mat Hyb_I;                   // ���̺긮�� �̹��� ��Ʈ���� ����

    for (int i = 0; i < 8; i++)
    {
        if (i >= 0 && i <= N1)
        { // N1���� Laplacian image
            Hybrid_Material.push_back(Laplacian_Pyramid_Vector_1[i]);
        }
        else if (i >= (8 - N2))
        { // N2���� Laplacian image + Last Gaussian image
            Hybrid_Material.push_back(Laplacian_Pyramid_Vector_2[i]);
        }
    }

    reverse(Hybrid_Material.begin(), Hybrid_Material.end());

    Hyb_I = Make_Hybrid_Func(Hybrid_Material);

    imwrite("Hybrid.jpg", Hyb_I);
    imshow("Hybrid", Hyb_I);
    waitKey(0);
}