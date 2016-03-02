#include "stdafx.h"
#include "Operater.h"


bool isLinear(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c){
	cv::Point2f a1 = b-a;
	double alen = b * a;
	if(alen <= 0.5)
		return true;
	a1.x /= alen ; a1.y /= alen;
	cv::Point2f a2 = c-a;
	double alen2 = c * a;
	if(alen2 <= 0.5)
		return true;
	a2.x /= alen2 ; a2.y /= alen2;
	cv::Vec3d cc = VecCross(cv::Vec3d(a1.x,a1.y,0) , cv::Vec3d(a2.x,a2.y,0));
	double len = abs(cc[2]);
	//std::cout<<"len = "<<len<<endl;
	if(len <= 0.02)
		return true;
	return false;

}



void sortPoints(std::vector<cv::Point2f>& pts){
	int pSize = pts.size();
	cv::Point2f LeftPoint = pts[pSize-1];
	cv::Point2f RightPoint = pts[pSize-2];

	if(LeftPoint.x > RightPoint.x){
		swapPoint(LeftPoint , RightPoint);
	}

	int mSize = pSize / 2;
	cv::Point2f mPoint = pts[mSize];

	int caseFlag = 1;
	if(mPoint.y < std::min(LeftPoint.y , RightPoint.y)){
		caseFlag = 2;
	}

	std::vector<cv::Point2f> newpts;
	std::vector<int>visited(pSize-2, 0);
	if(caseFlag == 1){
		newpts.push_back(RightPoint);
	}
	else{
		newpts.push_back(LeftPoint);
	}
	for(int i=0;i<pSize-2;i++){
		double dis = 9999.0;
		int uind = 0;
		cv::Point2f pTop = newpts[newpts.size()-1];
		for(int j=0;j<pSize-2;j++){
			if(!visited[j]){
				double dis_j = pts[j] * pTop;
				if(dis > dis_j){
					dis = dis_j;
					uind = j;
				}
			}
		}
		newpts.push_back(pts[uind]);
		visited[uind] = 1;
	}
	if(caseFlag == 1){
		newpts.push_back(LeftPoint);
	}
	else{
		newpts.push_back(RightPoint);
	}

	pts.clear();
	pts = newpts;

}

COperater::COperater(void)
{
}


COperater::~COperater(void)
{
}
