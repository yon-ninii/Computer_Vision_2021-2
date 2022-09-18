#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>       // Mat class        data structure          ƾ        ϴ
#include <opencv2/highgui/highgui.hpp> // GUI       õ    Ҹ       ϴ      imshow
#include "opencv2/imgproc/imgproc.hpp" //       ̹    ó    Լ         ϴ
using namespace std;
using namespace cv;
int myKernelConv9x9(uchar *arr, int kernel[][9], int x, int y, int width, int height, int ch, int p = 0)
{ //ä ΰ , 3ä    ϶   ȼ             p
    int sum = 0;
    int sumKernel = 0;

    for (int j = -4; j <= 4; j++)
    {
        for (int i = -4; i <= 4; i++)
        {
            if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
            {
                //     ũ      ڸ          ϱ
                if (ch == 1) // 1ä    ϶
                    sum += arr[(y + j) * width + (x + i)] * kernel[i + 4][j + 4];
                else if (ch == 3) // 3ä    ϶
                    sum += arr[(y + j) * width * 3 + (x + i) * 3 + p] * kernel[i + 4][j + 4];
                sumKernel += kernel[i + 4][j + 4];
            }
        }
    }
    if (sumKernel != 0)
    {
        return sum / sumKernel;
    } //      1       ȭ  ǵ                ⺯ȭ
    else
    {
        return sum;
    }
}

Mat myGaussianFilter(Mat srcImg, int ch)
{
    int width = srcImg.cols;
    int height = srcImg.rows;
    int kernel[9][9] = {1, 3, 5, 7, 10, 7, 5, 3, 1, //            9x9       ٻ簪       ġ Ͽ   .
                        3, 13, 16, 19, 22, 19, 16, 13, 3,
                        5, 16, 25, 28, 31, 28, 25, 16, 5,
                        7, 19, 28, 34, 37, 34, 28, 19, 7,
                        10, 22, 31, 37, 40, 37, 31, 22, 10,
                        7, 19, 28, 34, 37, 34, 28, 19, 7,
                        5, 16, 25, 28, 31, 28, 25, 16, 5,
                        3, 13, 16, 19, 22, 19, 16, 13, 3,
                        1, 3, 5, 7, 10, 7, 5, 3, 1};
    Mat dstImg_1(srcImg.size(), CV_8UC1); //1ä ο              size   1 ä               Ѵ .
    Mat dstImg_3(srcImg.size(), CV_8UC3); //3ä ο              size   3 ä               Ѵ .
    // Mat  ü       Ű
    uchar *srcData = srcImg.data;
    uchar *dstData_1 = dstImg_1.data;
    uchar *dstData_3 = dstImg_3.data;
    if (ch == 1)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                dstData_1[y * width + x] = myKernelConv9x9(srcData, kernel, x, y, width, height, 1);
                //  ռ         convolution        ũ  迭    Է
            }
        }
    }
    else if (ch == 3)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                dstData_3[y * width * 3 + x * 3] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 0);
                dstData_3[y * width * 3 + x * 3 + 1] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 1);
                dstData_3[y * width * 3 + x * 3 + 2] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 2);
                //  ռ         convolution        ũ  迭    Է
            }
        }
    }
    if (ch == 1)
    {
        return dstImg_1;
    }
    else if (ch == 3)
    {
        return dstImg_3;
    }
}

Mat doDft(Mat srcImg)
{
    Mat floatImg;
    srcImg.convertTo(floatImg, CV_32F);

    Mat complexImg;
    dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);
    return complexImg;
}

Mat getMagnitude(Mat complexImg)
{
    Mat planes[2];
    split(complexImg, planes);
    // Ǽ           и

    Mat magImg;
    magnitude(planes[0], planes[1], magImg);
    magImg += Scalar::all(1);
    log(magImg, magImg);
    //magnitude
    //log(1 + sqrt(Re(DFT(I))^) + Im(DFT(I))^2))

    return magImg;
}

Mat getPhase(Mat complexImg)
{
    Mat planes[2];
    split(complexImg, planes);
    // Ǽ           и

    Mat phaImg;
    phase(planes[0], planes[1], phaImg);
    //phase

    return phaImg;
}

Mat myNormalize(Mat src)
{
    Mat dst;
    src.copyTo(dst);
    normalize(dst, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC1);

    return dst;
}

Mat padding(Mat img)
{
    int dftRows = getOptimalDFTSize(img.rows);
    int dftCols = getOptimalDFTSize(img.cols);
    //DFT   꿡     ȭ

    Mat padded;
    copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));
    //DFT   꿡     ȭ          ǵ    padding

    return padded;
}

Mat centralize(Mat complex)
{
    Mat planes[2];
    split(complex, planes);
    int cx = planes[0].cols / 2;
    int cy = planes[1].rows / 2;

    Mat q0Re(planes[0], Rect(0, 0, cx, cy));
    Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
    Mat q2Re(planes[0], Rect(0, cy, cx, cy));
    Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

    Mat tmp;
    q0Re.copyTo(tmp);
    q3Re.copyTo(q0Re);
    tmp.copyTo(q3Re);
    q1Re.copyTo(tmp);
    q2Re.copyTo(q1Re);
    tmp.copyTo(q2Re);

    Mat q0Im(planes[1], Rect(0, 0, cx, cy));
    Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
    Mat q2Im(planes[1], Rect(0, cy, cx, cy));
    Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

    q0Im.copyTo(tmp);
    q3Im.copyTo(q0Im);
    tmp.copyTo(q3Im);
    q1Im.copyTo(tmp);
    q2Im.copyTo(q1Im);
    tmp.copyTo(q2Im);

    Mat centerComplex;
    merge(planes, 2, centerComplex);

    return centerComplex;
}

Mat setComplex(Mat magImg, Mat phaImg)
{
    exp(magImg, magImg);
    magImg -= Scalar::all(1);
    //magnitude        ݴ

    Mat planes[2];
    polarToCart(magImg, phaImg, planes[0], planes[1]);
    //     ǥ   ->        ǥ   (       ũ  κ    2       ǥ)

    Mat complexImg;
    merge(planes, 2, complexImg);
    // Ǽ   ,       merge

    return complexImg;
}

Mat doIdft(Mat complexImg)
{
    Mat idftcvt;
    idft(complexImg, idftcvt);
    //IDFT    ̿

    Mat planes[2];
    split(idftcvt, planes);

    Mat dstImg;
    magnitude(planes[0], planes[1], dstImg);
    normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
    dstImg.convertTo(dstImg, CV_8UC1);
    // Ϲ         type   ǥ           ȯ

    return dstImg;
}

Mat doLPF(Mat srcImg)
{
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);

    // <LPF>
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
    circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

    Mat magImg2;
    multiply(magImg, maskImg, magImg2);

    // <IDFT>
    normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);

    return myNormalize(dstImg);
}

Mat doHPF(Mat srcImg)
{
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);

    // <HPF>
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    Mat maskImg = Mat::ones(magImg.size(), CV_32F);
    circle(maskImg, Point(maskImg.cols / 2, (maskImg.rows / 2) - 5), 20, Scalar::all(0), -1, -1, 0);
    imshow("black", maskImg);
    waitKey(0);
    Mat magImg2;
    multiply(magImg, maskImg, magImg2);
    imshow("black", magImg2);
    waitKey(0);
    // <IDFT>
    normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);

    return myNormalize(dstImg);
}

Mat doBPF(Mat srcImg)
{
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    imshow("fasdasdfasdf", magImg);
    Mat phaImg = getPhase(centerComplexImg);

    // <LPF>
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    Mat mask_bImg = Mat::zeros(magImg.size(), CV_32F);
    Mat mask_sImg = Mat::zeros(magImg.size(), CV_32F);
    circle(mask_bImg, Point(mask_bImg.cols / 2, mask_bImg.rows / 2), 40, Scalar::all(1), -1, -1, 0);
    circle(mask_sImg, Point(mask_sImg.cols / 2, mask_sImg.rows / 2), 30, Scalar::all(1), -1, -1, 0);

    Mat bandmask_Img;
    subtract(mask_bImg, mask_sImg, bandmask_Img);
    imshow("band", bandmask_Img);
    waitKey(0);
    Mat magImg2;
    multiply(bandmask_Img, magImg, magImg2);
    imshow("bandbnad", magImg2);
    waitKey(0);
    // <IDFT>
    normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);

    return myNormalize(dstImg);
}
Mat deFlickering(Mat srcImg)
{
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);

    // <masking>
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    Mat maskImg = Mat::ones(magImg.size(), CV_32F);

    line(maskImg, Point(maskImg.cols / 2, 280), Point(maskImg.cols / 2, (maskImg.rows / 2) - 15), (0, 0, 0), 15);
    line(maskImg, Point(maskImg.cols / 2, (maskImg.rows / 2) + 15), Point(maskImg.cols / 2, maskImg.rows - 280), (0, 0, 0), 15);
    imshow("Y_sobelmask", maskImg);
    waitKey(0);

    Mat magImg2;
    multiply(magImg, maskImg, magImg2);
    imshow("line", magImg2);
    waitKey(0);
    // <IDFT>
    normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);

    return myNormalize(dstImg);
}

int myKernelConv3x3(uchar *arr, int kernel[][3], int x, int y, int width, int height)
{
    int sum = 0;
    int sumKernel = 0;

    // Ư   ȭ          ̿ ȭ ҿ           ϵ     ݺ
    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
            {
                sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
                sumKernel += kernel[i + 1][j + 1];
            }
        }
    }
    if (sumKernel != 0)
    {
        return sum / sumKernel;
    } //      1       ȭ  ǵ                ⺯ȭ
    else
    {
        return sum;
    }
    //color channel indexing       ٸ    ä κ
}

Mat mySobelFilter(Mat srcImg, int sel)
{
    //45
    int kernelX[3][3] = {-1, 0, 1,
                         -2, 0, 2,
                         -1, 0, 1};
    //135
    int kernelY[3][3] = {1, 2, 1,
                         0, 0, 0,
                         -1, -2, -1};

    Mat dstImg(srcImg.size(), CV_8UC1);
    uchar *srcData = srcImg.data;
    uchar *dstData = dstImg.data;
    int width = srcImg.cols;
    int height = srcImg.rows;

    if (sel == 2)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
                                          abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) /
                                         2;
            }
        }
    }
    return dstImg;
}

Mat myFreqSobelFilter_Y(Mat srcImg)
{
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);

    // <masking>
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
    //Mat mask_dImg = Mat::zeros(magImg.size(), CV_32F);
    line(maskImg, Point(maskImg.cols / 2, 0), Point(maskImg.cols / 2, (maskImg.rows / 2) - 50), Scalar::all(1), 50);
    line(maskImg, Point(maskImg.cols / 2, (maskImg.rows / 2) + 50), Point(maskImg.cols / 2, maskImg.rows), Scalar::all(1), 50);
    imshow("Y_sobelmask", maskImg);
    waitKey(0);
    //Mat line_img;
    //multiply(mask_uImg, mask_dImg, line_img);

    Mat magImg2;
    multiply(magImg, maskImg, magImg2);
    imshow("Y_masked", magImg2);
    waitKey(0);
    // <IDFT>
    normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);

    return myNormalize(dstImg);
}

Mat myFreqSobelFilter_X(Mat srcImg)
{
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);

    // <masking>
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
    //Mat mask_dImg = Mat::zeros(magImg.size(), CV_32F);
    line(maskImg, Point(0, maskImg.rows / 2), Point((maskImg.cols / 2) - 70, maskImg.rows / 2), Scalar::all(1), 100);
    line(maskImg, Point((maskImg.cols / 2) + 70, maskImg.rows / 2), Point(maskImg.cols, maskImg.rows / 2), Scalar::all(1), 100);
    imshow("X_sobelmask", maskImg);
    waitKey(0);
    //Mat line_img;
    //multiply(mask_uImg, mask_dImg, line_img);

    Mat magImg2;
    multiply(magImg, maskImg, magImg2);
    imshow("X_masked", magImg2);
    waitKey(0);
    // <IDFT>
    normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);

    return myNormalize(dstImg);
}

int main()
{
    Mat srcImg, dstImg, complexImg, centercomplexImg, magImg, phaImg, norm_magImg, norm_phaImg;

    // 1 mag
    srcImg = imread("img1.jpg", 0);
    complexImg = doDft(srcImg);
    magImg = getMagnitude(complexImg);
    dstImg = myNormalize(magImg);
    imshow("magnitude", dstImg);
    waitKey(0);

    // 2 pha
    srcImg = imread("img1.jpg", 0);
    complexImg = doDft(srcImg);
    phaImg = getPhase(complexImg);
    dstImg = myNormalize(phaImg);
    imshow("phase", dstImg);
    waitKey(0);

    // 3 centralize
    srcImg = imread("img1.jpg", 0);
    complexImg = doDft(srcImg);
    centercomplexImg = centralize(complexImg);
    magImg = getMagnitude(centercomplexImg);
    dstImg = myNormalize(magImg);
    imshow("centralize mag", dstImg);
    waitKey(0);

    //4 idft
    srcImg = imread("img1.jpg", 0);
    magImg = getMagnitude(complexImg);
    norm_magImg = myNormalize(magImg);
    phaImg = getPhase(complexImg);
    norm_phaImg = myNormalize(phaImg);
    imshow("mag", norm_magImg);
    waitKey(0);
    imshow("pha", norm_phaImg);
    waitKey(0);
    complexImg = setComplex(magImg, phaImg);
    dstImg = doIdft(complexImg);
    imshow("recovered", dstImg);
    waitKey(0);

    //5 HPF, LPF
    srcImg = imread("img1.jpg", 0);
    dstImg = doHPF(srcImg);
    imshow("HPF", dstImg);
    waitKey(0);
    dstImg = doLPF(srcImg);
    imshow("LPF", dstImg);
    waitKey(0);
    dstImg = doBPF(srcImg);
    imshow("BPF", dstImg);
    waitKey(0);

    //spatial domain sobel
    Mat sobel_hor, sobel_ver, sobel_mer;
    srcImg = imread("img2.jpg", 0);
    sobel_hor = mySobelFilter(srcImg, 0);
    imshow("hor", sobel_hor);
    waitKey(0);
    sobel_ver = mySobelFilter(srcImg, 1);
    imshow("ver", sobel_ver);
    waitKey(0);

    Mat padImg = padding(sobel_hor);
    complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    magImg = getMagnitude(centerComplexImg);
    phaImg = getPhase(centerComplexImg);
    norm_magImg = myNormalize(magImg);
    imshow("mag_hor", norm_magImg);
    waitKey(0);
    complexImg = setComplex(magImg, phaImg);

    padImg = padding(sobel_ver);
    complexImg = doDft(padImg);
    centerComplexImg = centralize(complexImg);
    magImg = getMagnitude(centerComplexImg);
    norm_magImg = myNormalize(magImg);
    norm_phaImg = myNormalize(phaImg);
    imshow("mag_ver", norm_magImg);
    waitKey(0);
    phaImg = getPhase(centerComplexImg);
    complexImg = setComplex(magImg, phaImg);

    //freq domain sobel
    sobel_hor, sobel_ver, sobel_mer;
    srcImg = imread("img2.jpg", 0);
    sobel_hor = myFreqSobelFilter_Y(srcImg);
    imshow("hor", sobel_hor);
    waitKey(0);
    sobel_ver = myFreqSobelFilter_X(srcImg);
    imshow("ver", sobel_ver);
    waitKey(0);

    // deflickering
    srcImg = imread("img3.JPG", 0);
    imshow("origin", srcImg);
    waitKey(0);

    Mat gauImg = myGaussianFilter(srcImg, 1);
    imshow("gau", gauImg);
    waitKey(0);

    dstImg = deFlickering(gauImg);
    imshow("recovered", dstImg);
    waitKey(0);

    return 0;
}