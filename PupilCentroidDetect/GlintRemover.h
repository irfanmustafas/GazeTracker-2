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

	//ͼ��Ԥ�������
	int PreProcess();

	//��߼��ȥ������
	int Process();

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

