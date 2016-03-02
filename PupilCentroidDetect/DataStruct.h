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

struct edgePoint{

	int ind;
	cv::Point2f pos;
	double angle;
public:
	edgePoint(const int& _ind , const cv::Point2f& _pos, const double& _angle): ind(_ind),
		pos(_pos) , angle(_angle){

	}
	edgePoint(){}
};

class CDataStruct
{
public:
	CDataStruct(void);
	~CDataStruct(void);
};

