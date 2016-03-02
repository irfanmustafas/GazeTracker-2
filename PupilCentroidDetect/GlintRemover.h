#pragma once
//Author: wishchin yang 20160301

#include <iostream>
#include <strstream>
 #include <fstream>
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

#include "D2DetectInPic.h"

class CGlintRemover
{
public:
	CGlintRemover(void);
	CGlintRemover(IplImage*  src);
	CGlintRemover(cv::Mat   &src);

	CGlintRemover(IplImage*  src,IplImage*  dst);
	CGlintRemover(cv::Mat   &src,cv::Mat   &dst);

	~CGlintRemover(void);
	int DetectArc (int argc, _TCHAR* argv[]);
	int DetectArc (cv::Mat &ImageColor);
	int DetectArcRs (cv::Mat &ImageColor);

public:

	//图像预处理过程
	int PreProcess();

	//光斑检测去除过程
	int Process();


	//获取图像结果-获取结果图像
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
	//hough圆检测
	cv::Mat  CircleCheckHough(cv::Mat  &src);

	//使用固定值去除黑边
	cv::Mat  CutBlackBlock(cv::Mat  &src,CvPoint  &center, int radius);

	//使用边缘检测进行测试
	cv::Mat  EdgeDetectBlock(cv::Mat  &src,CvPoint  &center, int radius);

	//去除边缘的疙瘩，把光斑去除放在此过程中
	int WapeOutEdgeLump();

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
	double mThresholdInvLowerBound; //光斑亮度值的下界
	double mThresholdInvHigherBound;//光斑亮度值的上界

	double mThresholdPupilHigherBound; //光斑亮度值的上界
	double mThresholdPupilLowerBound;  //光斑亮度值的下界
	double mThresholdPupilInvHigherBound; //光斑亮度值的反转下界
	double mThresholdPupilInvLowerBound; //光斑亮度值的反转下界


	//检测弧的大小的上下界
	double  mThresholdEdgePointsLowerBound;
	double mThresholdEdgePointsHigherBound;

private:
	//snip..........
	IplImage*  mImageOut;
	cv::Mat      mMatOut;

	int mImageHeight;
	int  mImageWidth;

	CD2DetectInPic mLineDetector;

public:
	double DrawHist(cv::Mat &histogram);
	double DrawHist(double hh[256]);

};

