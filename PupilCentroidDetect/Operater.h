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

#include "DataStruct.h"

inline double operator * (const edgePoint& a, const edgePoint& b)
{
	return sqrt((double)(a.pos.x - b.pos.x) * (a.pos.x - b.pos.x) + 
		(a.pos.y - b.pos.y) * (a.pos.y - b.pos.y));
}

inline double operator * (const cv::Point2f& a, const cv::Point2f& b)
{
	return sqrt((double)(a.x - b.x) * (a.x - b.x) + 
		(a.y - b.y) * (a.y - b.y));
}

inline cv::Point2f operator - (const cv::Point2f& a, const cv::Point2f& b)
{
	return cv::Point2f(a.x-b.x,a.y-b.y);
}

inline bool cmp(edgePoint a, edgePoint b)
{
	return a.angle > b.angle;
}

inline cv::Vec3d VecCross(const cv::Vec3d& p1, const cv::Vec3d& p2)
{
	return cv::Vec3d(
		p1[1]*p2[2]-p1[2]*p2[1],p1[2]*p2[0]-p1[0]*p2[2],
		p1[0]*p2[1]-p1[1]*p2[0]
	);
}

inline void swapPoint(cv::Point2f& a, cv::Point2f& b){
	cv::Point2f c;
	c = b; b= a; a = c;
}

bool isLinear(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c);
void sortPoints(std::vector<cv::Point2f>& pts);

class COperater
{
public:
	COperater(void);
	~COperater(void);
};

