#pragma once
//Author: wishchin yang 20160301
//ͫ�׵Ĺ���Ϊһ���ڶ�������Ϊһ��Ȧ,Ϊ�ڶ�ģ�Ͷ��Ƿ���ģ��
//    ͫ�����Ķ��壺����   ���㷽�������Ĺ�ʽ
//    ������̣� �ձ�Ե������ͨ�򣬼�����ͨ�������

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

class CParamPupil
{
public:
	CParamPupil(void);
	~CParamPupil(void);

public:
	cv::Point PCenter;
	std::vector<cv::Point>   PEdge;
	std::vector<cv::Point>   PArea;

protected:
	//����߳�
	int CalEdgeLength(std::vector<cv::Point>  &edge);
	//����ߵ�����
	int CalEdgeCentroid(std::vector<cv::Point>  &edge,cv::Point &centroid);
	//������ͨ��
	int CalConnectRegion(std::vector<cv::Point>  &edge,std::vector<cv::Point>  &region);
	//������ͨ������
	int CalConnectRegion(std::vector<cv::Point>  &region, cv::Point &centroid);

private:
	//ͫ�׵���������
	cv::Point mPCentroid;//��������
	std::vector<cv::Point>   mPEdge;//�����Ե
	std::vector<cv::Point>   mPArea;//����ڶ�����ͨ��

};

