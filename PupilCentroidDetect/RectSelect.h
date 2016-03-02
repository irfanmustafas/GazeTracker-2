#pragma once
//Author: wishchin yang 20160301
//辅助深度学习的数据集选择，为一个图像选取选取多个框

#define _USE_MATH_DEFINES   //　看math.h中的定义  
//#include <math.h>
#include <iostream>
#include <strstream>
#include <cmath>

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

class CRectSelect
{
public:
	CRectSelect(void);
	CRectSelect(cv::Mat &Image);
	CRectSelect(int width,int height);

	~CRectSelect(void);

public:
	//snip..........
	//选取多个框
	//选取移动框
	int SelectMoveRect();
	//选取旋转框
	int SelectRoateRect();
	int SelectRoateRect(int radius);
	int SelectRoateRect(int radius,CvPoint  &pupilCenter);
	//选取旋转缩放框
	int SelectRoateResizeRect();
	//选取旋转缩放移动框
	int SelectRoateResizeMoveRect();


	//选取旋转框框定的图像
	//生成5个旋转矩阵，存储在 this->mImageRectRoate 里面
	//同时生成5个二倍大小的矩阵，存储在mImageRect2 里面
	int SelectRoatePatch(int radius,CvPoint  &pupilCenter);

	//生成10个图片，存储在mImageRectRoateResize 里面
	int SelectRoateResizePatch(int radius,CvPoint  &pupilCenter);
	
	//选取旋转缩放移动框图片
	//生成90个图片，存储在mImageRectRoateResizeMove 里面
	int SelectRoateResizeMovePatch(int radius,CvPoint  &pupilCenter);
	

	//获取图片块
	int GetPicPatches();
	int GetPicPatches(std::vector<cv::Mat> &patchList);

private:
	//snip..........

	int mNumTotalRect;           //获取移动框的个数

	//int mNumMoveRect;            //获取移动框的个数
	int mNumRoateRect;           //获取旋转框的个数
	int mNumRoateResizeRect;     //获取旋转后缩放框的个数
	int mNumRoateResizeMoveRect; //获取旋转后缩放移动框的个数

	//int mNumRoateMoveRect; //获取旋转后移动框的个数

private:
	//snip..........
	double  mRadiusS;//最终搜索半径，即是正方形框的半边长。
	double  mRadius;//旋转半径，即是正方形框的半边长。

	double  mDetaAngle;           //旋转间隔

	double  mDetaMoveStep;           //移动单位个数
	int     mDetaMovePixels;           //移动像素间隔
	int     mDetaMovePixelsX;           //移动像素间隔
	int     mDetaMovePixelsY;           //移动像素间隔

	int  mWidth;//载入图像的宽和高
	int mHeight;
	int  mRectWidth;//载入图像的宽和高
	int mRectHeight;

	//设定的缩放因子
	double mZoomFactor;

private:
	//snip..........
	//手动获取的瞳孔中心的坐标
	cv::Mat  mImage;
	CvPoint  mPupilCenter;

	//获取框的坐标
	CvRect  mNormalRect;
	CvRect  mDoubleRect;
	std::vector<CvRect>  mRectRoate;
	std::vector<CvRect>  mRectRoateResize;
	std::vector<CvRect>  mRectRoateResizeMove;

	//获取框内的图像
	std::vector<cv::Mat>  mImageRect;//使用图像的局部，想要选取框的2倍
	
	std::vector<cv::Mat>  mImageRectRoate;
	std::vector<cv::Mat>  mImageRectRoate2;//使用图像的局部，想要选取框的2倍
	std::vector<cv::Mat>  mImageRectRoateResize;
	std::vector<cv::Mat>  mImageRectRoateResize2;//生成10个大2倍的图像块
	std::vector<cv::Mat>  mImageRectRoateResizeMove;
	std::vector<cv::Point>  mCenterRoateResizeMove;

private:
	bool RerankXY(cv::Mat &rectM, cv::Mat &rectNormal,CvPoint &pStart);
	//int CPlot::cvDrawCrossCursor(
	int CvDrawCrossCursor(
		cv::Mat &Canvas,cv::Point &Center,int Length,cv::Scalar &Color,int Width,int CV_A_Type,int Mark);

};

