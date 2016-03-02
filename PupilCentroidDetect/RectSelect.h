#pragma once
//Author: wishchin yang 20160301
//�������ѧϰ�����ݼ�ѡ��Ϊһ��ͼ��ѡȡѡȡ�����

#define _USE_MATH_DEFINES   //����math.h�еĶ���  
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
	//ѡȡ�����
	//ѡȡ�ƶ���
	int SelectMoveRect();
	//ѡȡ��ת��
	int SelectRoateRect();
	int SelectRoateRect(int radius);
	int SelectRoateRect(int radius,CvPoint  &pupilCenter);
	//ѡȡ��ת���ſ�
	int SelectRoateResizeRect();
	//ѡȡ��ת�����ƶ���
	int SelectRoateResizeMoveRect();


	//ѡȡ��ת��򶨵�ͼ��
	//����5����ת���󣬴洢�� this->mImageRectRoate ����
	//ͬʱ����5��������С�ľ��󣬴洢��mImageRect2 ����
	int SelectRoatePatch(int radius,CvPoint  &pupilCenter);

	//����10��ͼƬ���洢��mImageRectRoateResize ����
	int SelectRoateResizePatch(int radius,CvPoint  &pupilCenter);
	
	//ѡȡ��ת�����ƶ���ͼƬ
	//����90��ͼƬ���洢��mImageRectRoateResizeMove ����
	int SelectRoateResizeMovePatch(int radius,CvPoint  &pupilCenter);
	

	//��ȡͼƬ��
	int GetPicPatches();
	int GetPicPatches(std::vector<cv::Mat> &patchList);

private:
	//snip..........

	int mNumTotalRect;           //��ȡ�ƶ���ĸ���

	//int mNumMoveRect;            //��ȡ�ƶ���ĸ���
	int mNumRoateRect;           //��ȡ��ת��ĸ���
	int mNumRoateResizeRect;     //��ȡ��ת�����ſ�ĸ���
	int mNumRoateResizeMoveRect; //��ȡ��ת�������ƶ���ĸ���

	//int mNumRoateMoveRect; //��ȡ��ת���ƶ���ĸ���

private:
	//snip..........
	double  mRadiusS;//���������뾶�����������ο�İ�߳���
	double  mRadius;//��ת�뾶�����������ο�İ�߳���

	double  mDetaAngle;           //��ת���

	double  mDetaMoveStep;           //�ƶ���λ����
	int     mDetaMovePixels;           //�ƶ����ؼ��
	int     mDetaMovePixelsX;           //�ƶ����ؼ��
	int     mDetaMovePixelsY;           //�ƶ����ؼ��

	int  mWidth;//����ͼ��Ŀ�͸�
	int mHeight;
	int  mRectWidth;//����ͼ��Ŀ�͸�
	int mRectHeight;

	//�趨����������
	double mZoomFactor;

private:
	//snip..........
	//�ֶ���ȡ��ͫ�����ĵ�����
	cv::Mat  mImage;
	CvPoint  mPupilCenter;

	//��ȡ�������
	CvRect  mNormalRect;
	CvRect  mDoubleRect;
	std::vector<CvRect>  mRectRoate;
	std::vector<CvRect>  mRectRoateResize;
	std::vector<CvRect>  mRectRoateResizeMove;

	//��ȡ���ڵ�ͼ��
	std::vector<cv::Mat>  mImageRect;//ʹ��ͼ��ľֲ�����Ҫѡȡ���2��
	
	std::vector<cv::Mat>  mImageRectRoate;
	std::vector<cv::Mat>  mImageRectRoate2;//ʹ��ͼ��ľֲ�����Ҫѡȡ���2��
	std::vector<cv::Mat>  mImageRectRoateResize;
	std::vector<cv::Mat>  mImageRectRoateResize2;//����10����2����ͼ���
	std::vector<cv::Mat>  mImageRectRoateResizeMove;
	std::vector<cv::Point>  mCenterRoateResizeMove;

private:
	bool RerankXY(cv::Mat &rectM, cv::Mat &rectNormal,CvPoint &pStart);
	//int CPlot::cvDrawCrossCursor(
	int CvDrawCrossCursor(
		cv::Mat &Canvas,cv::Point &Center,int Length,cv::Scalar &Color,int Width,int CV_A_Type,int Mark);

};

