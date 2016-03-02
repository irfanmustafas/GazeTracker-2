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

	//ͼ��Ԥ�������
	int PreProcess();

	//��߼��ȥ������
	int Process();


	//��ȡͼ����-��ȡ���ͼ��
	int GetOutcome(IplImage*  dst);
	int GetOutcome(cv::Mat   &dst);

public:
	//snip..........

	//ȥ����Χ�ĺ�Ȧ��ʹ��houghԲ��⣬ȥ������Բ
	//ʹ����ֵ120�������
	int RemoveBackGround();

	//����㣬��С��ѡ�񣬸�����ֵ�������ȱ仯��
	int DetectGlint();

	//ȥ����㣬��С��ѡ�񣬸��ݴ�С�ж��Ƿ�ȥ��
	int RemoveGlint();

public:
	//houghԲ���
	cv::Mat  CircleCheckHough(cv::Mat  &src);

	//ʹ�ù̶�ֵȥ���ڱ�
	cv::Mat  CutBlackBlock(cv::Mat  &src,CvPoint  &center, int radius);

	//ʹ�ñ�Ե�����в���
	cv::Mat  EdgeDetectBlock(cv::Mat  &src,CvPoint  &center, int radius);

	//ȥ����Ե�ĸ�񣬰ѹ��ȥ�����ڴ˹�����
	int WapeOutEdgeLump();

public:
	//snip..........
	//������ֵ
	double SearchBrightnessPeak();

	//̽����ֵ��Χ�����һ����ֵ��ֵ
	double SearchThresholdLowerBound();

		//̽����ֵ��Χ�����һ����ֵ��ֵ
	double SearchThresholdHigherBound();


private:
	//snip..........

	double mBrightnessPeak;      //̽�����߷�ֵ��Ĭ������Ϊ���ֵ
	double mThresholdLowerBound; //�������ֵ���½�
	double mThresholdHigherBound;//�������ֵ���Ͻ�
	double mThresholdInvLowerBound; //�������ֵ���½�
	double mThresholdInvHigherBound;//�������ֵ���Ͻ�

	double mThresholdPupilHigherBound; //�������ֵ���Ͻ�
	double mThresholdPupilLowerBound;  //�������ֵ���½�
	double mThresholdPupilInvHigherBound; //�������ֵ�ķ�ת�½�
	double mThresholdPupilInvLowerBound; //�������ֵ�ķ�ת�½�


	//��⻡�Ĵ�С�����½�
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

