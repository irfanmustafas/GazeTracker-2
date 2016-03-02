#pragma once
//Author: wishchin yang 20160301
//图像和点云的二维探测：线的显著性检测
//    直线、曲线、多边形、圆的探测


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
#include "Operater.h"

class CD2DetectInPic
{

public:
	CD2DetectInPic(void);
	~CD2DetectInPic(void);

public:
	void getRectBottomLine(const cv::RotatedRect& r ,std::vector<cv::Point2f>& pts);
	double isCircle(const std::vector<cv::Point>& hull);

	//确定弧的弧的相似性质
	void sampleArc(std::vector<cv::Point2f>& arc_Points);

	//确定不封闭的弧
	bool findArc2(const std::vector<cv::Point>& hpts ,std::vector<cv::Point2f>& arc_pts,std::vector<int>& inds);

};

