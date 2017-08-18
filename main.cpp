// CalibrationADAS.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <algorithm>
#include "FindCorners.h "
#include <stdio.h>
#include <iostream>
#include <time.h>
using namespace cv;
using namespace std;
using std::vector;
vector<Point2i> points;

int _tmain(int argc, _TCHAR* argv[])
{
	//Mat kernels;
	//FileStorage fs2("templateA1.xml", FileStorage::READ);//读XML文件
	//fs2["templateA1"] >> kernels;
	//cout << "kernels: " << kernels << endl;

	//读入原始图像
	Mat src; //输入图像
	cout << "This is a demo for Parking slot detection." << endl;
	cout << "开始读入图像..." << endl;
	string filename = "Img\\02.png";//图像路径位置 "Img\\birdView0015.png"   calib\\_70.png
	src = imread(filename, -1);//载入测试图像
	if (src.empty())//不能读取图像
	{
		printf("Cannot read image file: %s\n", filename.c_str());
		return -1;
	}
	namedWindow("SrcImg");//创建窗口，显示原始图像
	imshow("SrcImg", src);

	vector<Point> corners;//存储找到的角点
	FindCorners corner_detector(src);
	corner_detector.detectCorners(src, corners,0.025);
	return 0;
}

