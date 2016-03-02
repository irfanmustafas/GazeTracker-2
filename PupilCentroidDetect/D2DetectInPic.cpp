#include "stdafx.h"
#include "D2DetectInPic.h"

using namespace std;
using namespace  cv;

CD2DetectInPic::CD2DetectInPic(void)
{
}


CD2DetectInPic::~CD2DetectInPic(void)
{
}

void CD2DetectInPic::getRectBottomLine(const RotatedRect& r , vector<Point2f>& pts)
{

	//vector<Point2f>pts;
	Point2f * pp = new Point2f(4);
	r.points(pp);
	double mid_x = 0;
	for(int i=0;i<4;i++){
		mid_x += pp[i].x;
	}
	mid_x /= 4;

	double lmax = -99999, rmax = -99999;
	int lu = 0, ru = 0;
	for(int i=0;i<4;i++){
		if(pp[i].x < mid_x){
			if(lmax < pp[i].y ){
				lmax = pp[i].y;
				lu = i;
			}
		}
		else if(pp[i].x >= mid_x){
			if(rmax < pp[i].y){
				rmax = pp[i].y;
				ru = i;
			}
		}
	}

	pts.push_back(pp[lu]);
	pts.push_back(pp[ru]);


}

double CD2DetectInPic::isCircle(const vector<Point>& hull)
{

	RotatedRect r = minAreaRect(hull);
	vector<Point2f>pts;
	getRectBottomLine(r , pts);
	Point2f center = Point2f((pts[0].x + pts[1].x) / 2.0,(pts[0].y + pts[1].y) / 2.0);
	double mean_v = 0;
	vector<double>lens(hull.size());
	for(int i=0;i<hull.size();i++){
		double tlen = sqrt((hull[i].x - center.x) * (hull[i].x - center.x) + 
			(hull[i].y - center.y) * (hull[i].y - center.y));
		lens[i] = tlen;
		mean_v += tlen;
	}
	mean_v /= hull.size();
	double sigma = 0.0;
	for(int i=0;i<hull.size();i++){
		sigma += (lens[i] - mean_v) * (lens[i] - mean_v);
	}
	sigma /= hull.size();
	sigma = sqrt(sigma);
	return sigma;

}


//确定弧的弧的相似性质
void CD2DetectInPic::sampleArc(vector<Point2f>& arc_points)
{
	vector<Point2f> new_arc;
	vector<double> arc_dis;
	double max_dis = 99999; int uind = 0;
	for(int i=1;i<arc_points.size();i++){
		double tdis = arc_points[i] * arc_points[i-1];
		arc_dis.push_back(tdis);
		if(max_dis > tdis){
			max_dis = tdis;
			uind = i;
		}
	}
	for(int i=0;i<arc_dis.size();i++){
		int sampleNum = floor(arc_dis[i] / max_dis);
		Point2f n = arc_points[i+1] - arc_points[i];
		double nLen = arc_points[i+1] * arc_points[i];
		n.x /= nLen ; n.y /= nLen;
		for(int j=0;j<sampleNum;j++){
			Point2f tp = arc_points[i];
			tp.x += j * max_dis * n.x;
			tp.y += j * max_dis * n.y;
			new_arc.push_back(tp);
		}
		Point2f ltp = arc_points[i];
		ltp.x += sampleNum * max_dis * n.x;
		ltp.y += sampleNum * max_dis * n.y;
		double tLen = ltp * arc_points[i+1];
		if(tLen < 0.5 * max_dis){

		}
		else{
			new_arc.push_back(ltp);
		}
	}
	double lastDis = arc_points[arc_points.size()-1] * new_arc[new_arc.size()-1];
	if(lastDis < 0.5 * max_dis){
		new_arc[new_arc.size()-1] = arc_points[arc_points.size()-1];
	}
	else{
		new_arc.push_back(arc_points[arc_points.size()-1]);
	}
	arc_points.clear();
	arc_points = new_arc;
}


//确定不封闭的弧
bool CD2DetectInPic::findArc2(const vector<Point>& hpts , vector<Point2f>& arc_pts, vector<int>& inds)
{
	arc_pts.clear();
	if(hpts.size() < 2){
		return false;
	}
	vector<edgePoint>epts(hpts.size());

	for(int i=0;i<hpts.size();i++)
	{
		cv::Vec3d v1,v2,v3;
		if(i == 0){
			v1 = cv::Vec3d(hpts[0].x - hpts[hpts.size()-1].x,
				hpts[0].y - hpts[hpts.size()-1].y, 0);
			v2 = cv::Vec3d(hpts[1].x - hpts[0].x,
				hpts[1].y - hpts[0].y, 0);
		}
		else if(i == hpts.size() - 1){
			v1 = cv::Vec3d(hpts[i].x - hpts[i-1].x,
				hpts[i].y - hpts[i-1].y, 0);
			v2 = cv::Vec3d(hpts[0].x - hpts[i].x,
				hpts[0].y - hpts[i].y, 0);
		}
		else{
			v1 = cv::Vec3d(hpts[i].x - hpts[i-1].x,
				hpts[i].y - hpts[i-1].y, 0);
			v2 = cv::Vec3d(hpts[i+1].x - hpts[i].x,
				hpts[i+1].y - hpts[i].y, 0);
		}
		double len1 = sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
		double len2 = sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
		v1[0] /= len1; v1[1] /= len1; v1[2] /= len1;
		v2[0] /= len2; v2[1] /= len2; v2[2] /= len2;
		v3 = VecCross(v1,v2);
		epts[i].angle = sqrt(v3[0]*v3[0]+v3[1]*v3[1]+v3[2]*v3[2]);
		epts[i].ind = i;
		epts[i].pos = hpts[i];
	}

	sort(epts , cmp);
	inds.push_back(epts[0].ind);
	for(int i=1;i<epts.size();i++){
		double len = epts[i] * epts[0];
		if(len <= 10)
			continue;
		else{
			inds.push_back(epts[i].ind);
			break;
		}
	}
	if(inds.size() <= 1){
		return false;
	}
	//inds.push_back(epts[1].ind);

	int leftlen = max(inds[0] , inds[1]) - min(inds[0] , inds[1]);
	int rightlen = hpts.size() - leftlen;

	if ( inds[0]< hpts.size() && inds[1]< hpts.size())//附件！wishchin!!!排除大下标
	{
		if(leftlen > rightlen){
			for(int i=0;i<hpts.size();i++){
				if(i >= min(inds[0] , inds[1]) && i <= max(inds[0] , inds[1])){
					if(!isLinear(hpts[inds[0]] , hpts[inds[1]] , hpts[i])){
						arc_pts.push_back(hpts[i]);
					}
				}
			}
		}
		else{
			for(int i=0;i<hpts.size();i++){
				if(i > min(inds[0] , inds[1]) && i < max(inds[0] , inds[1])){
					//arc_pts.push_back(hpts[i]);
				}
				else{
					//std::cout<<"id = "<<i<<endl;
					if(!isLinear(hpts[inds[0]] , hpts[inds[1]] , hpts[i])){
						arc_pts.push_back(hpts[i]);
					}
				}
			}
		}
		arc_pts.push_back(hpts[inds[0]]);
		arc_pts.push_back(hpts[inds[1]]);

		return true;
	}
	else
		return false;
	
}