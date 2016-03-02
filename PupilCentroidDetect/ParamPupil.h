#pragma once
//Author: wishchin yang 20160301
//瞳孔的构成为一个黑洞，本质为一个圈,为黑洞模型而非反射模型
//    瞳孔中心定义：质心   计算方法：质心公式
//    计算过程： 闭边缘内求连通域，计算连通域的质心

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
	//计算边长
	int CalEdgeLength(std::vector<cv::Point>  &edge);
	//计算边的质心
	int CalEdgeCentroid(std::vector<cv::Point>  &edge,cv::Point &centroid);
	//计算连通域
	int CalConnectRegion(std::vector<cv::Point>  &edge,std::vector<cv::Point>  &region);
	//计算连通域质心
	int CalConnectRegion(std::vector<cv::Point>  &region, cv::Point &centroid);

private:
	//瞳孔的自身性质
	cv::Point mPCentroid;//定义中心
	std::vector<cv::Point>   mPEdge;//定义边缘
	std::vector<cv::Point>   mPArea;//定义黑洞，连通域

};

