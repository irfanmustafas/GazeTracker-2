#pragma once

#include <iostream>
#include <strstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "cv.h" 
#include "highgui.h"  

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>


class CGlintRemover
{
public:
	CGlintRemover(void);
	CGlintRemover(IplImage*  src);
	CGlintRemover(cv::Mat   &src);

	CGlintRemover(IplImage*  src,IplImage*  dst);
	CGlintRemover(cv::Mat   &src,cv::Mat   &dst);

	~CGlintRemover(void);

public:

	//图像预处理过程
	int PreProcess();

	//光斑检测去除过程
	int Process();

	int GetOutcome(IplImage*  dst);
	int GetOutcome(cv::Mat   &dst);

public:
	//snip..........

	//去除周围的黑圈，使用hough圆检测，去掉最大的圆
	//使用中值120进行填充
	int RemoveBackGround();

	//检测光点，大小可选择，根据阈值调节亮度变化域
	int DetectGlint();

	//去除光点，大小可选择，根据大小判定是否去除
	int RemoveGlint();

public:
	//snip..........
	//检测光点峰值
	double SearchBrightnessPeak();

	//探测阈值范围，获得一个阈值极值
	double SearchThresholdLowerBound();

		//探测阈值范围，获得一个阈值极值
	double SearchThresholdHigherBound();

private:
	//snip..........

	double mBrightnessPeak;      //探测亮斑峰值，默认设置为最大值
	double mThresholdLowerBound; //光斑亮度值的下界
	double mThresholdHigherBound;//光斑亮度值的上界

private:
	//snip..........
	IplImage*  mImageOut;
	cv::Mat      mMatOut;

	int mImageHeight;
	int  mImageWidth;

public:
	double DrawHist(cv::Mat &histogram);
	double DrawHist(double hh[256]);

	cv::Mat  circleCheckHough(cv::Mat  &Src);
};

