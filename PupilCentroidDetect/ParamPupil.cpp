#include "stdafx.h"
#include "ParamPupil.h"


CParamPupil::CParamPupil(void)
{

}


CParamPupil::~CParamPupil(void)
{

}


//计算边长
int CParamPupil::CalEdgeLength(std::vector<cv::Point>  &edge)
{

	return 1;
}

//计算边的质心
int CParamPupil::CalEdgeCentroid(std::vector<cv::Point>  &edge,cv::Point &centroid)
{

	return 1;
}

//计算连通域
int CParamPupil::CalConnectRegion(std::vector<cv::Point>  &edge,std::vector<cv::Point>  &region)
{

	return 1;
}

//计算连通域质心
int CParamPupil::CalConnectRegion(std::vector<cv::Point>  &region, cv::Point &centroid)
{

	return 1;
}
