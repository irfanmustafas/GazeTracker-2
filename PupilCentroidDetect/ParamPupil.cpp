#include "stdafx.h"
#include "ParamPupil.h"


CParamPupil::CParamPupil(void)
{

}


CParamPupil::~CParamPupil(void)
{

}


//����߳�
int CParamPupil::CalEdgeLength(std::vector<cv::Point>  &edge)
{

	return 1;
}

//����ߵ�����
int CParamPupil::CalEdgeCentroid(std::vector<cv::Point>  &edge,cv::Point &centroid)
{

	return 1;
}

//������ͨ��
int CParamPupil::CalConnectRegion(std::vector<cv::Point>  &edge,std::vector<cv::Point>  &region)
{

	return 1;
}

//������ͨ������
int CParamPupil::CalConnectRegion(std::vector<cv::Point>  &region, cv::Point &centroid)
{

	return 1;
}
