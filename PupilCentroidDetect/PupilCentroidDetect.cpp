// PupilCentroidDetect.cpp : 定义控制台应用程序的入口点。
#include "stdafx.h"
#include <windows.h>

#include <iostream>
#include <strstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include<boost/thread.hpp>
#include<boost/filesystem.hpp>

#include "cv.h" 
#include "highgui.h"  

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "GlintRemover.h"
#include "RectSelect.h"

using namespace std;
using namespace  cv;

bool loadFileList(
	const boost::filesystem::path &base_dir, 
	const std::string &extension,  
	std::vector<std::string> &FileList);  

//int testSkin(int argc, char* argv[])  ;
//int cutBlack(cv::Mat  src,cv::Mat dst);
//int cutBlack(IplImage*  src, IplImage*  &dst);
//void cvSkinHSV(IplImage* src,IplImage* dst)  ;
//void cvSkinYUV(IplImage* src,IplImage* dst)  ;
//void cvSkinOtsu(IplImage* src, IplImage* dst) ;

void cvThresholdOtsu(IplImage* src, IplImage* dst)  ;

//// skin detection in rg space  
//void cvSkinRG(IplImage*  rgb,  IplImage* gray);
//void  SkinRGB(IplImage*  rgb,  IplImage* _dst);

//int testfingerTrack(int argc, char* argv[]);
//int cvOutlineFinger(IplImage* src,IplImage* dst);

int testKP(int argc, char* argv[]);
void cvFingerKeypoints(IplImage* src,IplImage* dst)  ;
void cvFingerKeypoints(cv::Mat  src,cv::Mat dst)  ;

//int testTrack(int argc, char* argv[]);
void cvCannyImg(IplImage* rgb, IplImage*  Edge);

//void testEigen(int argc, char* argv[]);
//void PrintMatrix(Eigen::MatrixXf  &M);
//void testFast(int argc, char* argv[]);

int testGlintRemover(int argc, char* argv[]);
int testRectSelector(int argc, char* argv[]);

int _tmain(int argc, char* argv[])
{
	//testSkin(argc,argv);
	//testfingerTrack(argc,argv);
	//testTrack(argc, argv);
	//testFast(argc, argv);
	
	//double T = atan2(-0.375,1);std::cout<< "TestAtan2: "<<  T  << std::endl;

	//testEigen(argc,argv);

	//testKP(argc,argv);

	//testRectSelector(argc,argv);

	testGlintRemover(argc,argv);

	return 0;
}

//void testEigen(int argc, char* argv[])
//{
//
//	Eigen::MatrixXf  M;
//	int rows =5;
//	int cols =4;
//
//	//M= 1.0/2* Eigen::MatrixXf::Random(rows,cols) + 1.0/2 *Eigen::MatrixXf::Ones(rows,cols) ;
//	//M=  Eigen::MatrixXf::Random(rows,cols)  ;
//	//PrintMatrix(M);
//
//	int nPointsRand = 12;
//	Eigen::MatrixXf X = Eigen::MatrixXf::Random(1,nPointsRand);
//	Eigen::MatrixXf Y = Eigen::MatrixXf::Random(1,nPointsRand);
//	Eigen::MatrixXf Z = Eigen::MatrixXf::Random(1,nPointsRand);
//	Eigen::MatrixXf theta  = Eigen::MatrixXf::Random(1,nPointsRand);
//	Eigen::MatrixXf phi    = Eigen::MatrixXf::Random(1,nPointsRand);
//	Eigen::MatrixXf lambda = Eigen::MatrixXf::Random(1,nPointsRand);
//	Eigen::MatrixXf randSphere6D =Eigen::MatrixXf (6,nPointsRand);//randSphere6D 是6*nPointsRandDouble！
//
//	//randSphere6D = [ X; Y; Z; theta; phi; lambda ];
//	//randSphere6D(0)=X(0);
//	//randSphere6D(1)=Y(0);
//	//randSphere6D(2)=Z(0);
//	//randSphere6D(3)= theta(0);
//	//randSphere6D(4)=   phi(0);
//	//randSphere6D(5)=lambda(0);
//	randSphere6D<< X,Y,Z,theta,phi,lambda;
//	PrintMatrix(randSphere6D);
//
//	//多维数组仍然按一维索引
//	Eigen::VectorXf Line1; Line1<<randSphere6D.row(0);
//	Eigen::VectorXf Line2= randSphere6D.row(1);
//	Eigen::VectorXf Line3= randSphere6D.row(2);
//	double N1 =atan(  randSphere6D( 0)*randSphere6D( 2)/ ( Line1.norm()* Line3.norm() )   ) *180/M_PI;
//	double N2 =atan(  randSphere6D( 1)*randSphere6D( 2)/ ( Line2.norm()* Line3.norm() )   ) *180/M_PI;
//
//	return ;
//}
//
//void PrintMatrix(Eigen::MatrixXf  &M)
//{
//	for (int i=0;i< M.rows();++i)
//	{
//		for (int j=0;j< M.cols();++j){
//			std::cout<<"    M(i,j): " << M(i,j);
//		}
//		std::cout<<std::endl;
//	}
//	
//	return ;
//}

////把黑色替换成蓝色，解决肤色检测问题
//int cutBlack(cv::Mat  src,cv::Mat dst)
//{
//
//	return 1;
//}
//
////把黑色替换成蓝色，解决肤色检测问题
////先Cut75，把RGB加和为75的换成blue！
//int cutBlack(IplImage*  rgb, IplImage*  &Rg)
//{
//	assert(rgb->nChannels==3 && Rg->nChannels==3);  
//
//	const int R=2;  
//	const int G=1;  
//	const int B=0;  
//
//
//	//IplImage* Rg =cvCreateImage(cvGetSize(rgb),8,3); 
//	//IplImage* Rg=NULL;
//	cvCopyImage(rgb,Rg);
//
//	double Aup=-1.8423;  
//	double Bup=1.5294;  
//	double Cup=0.0422;  
//	double Adown=-0.7279;  
//	double Bdown=0.6066;  
//	double Cdown=0.1766;  
//
//	for (int h=0;h<rgb->height;h++) {  
//		//unsigned char* pGray=(unsigned char*)gray->imageData+h*gray->widthStep;  
//		unsigned char* pRGB=(unsigned char* )rgb->imageData+h*rgb->widthStep;  
//		unsigned char* prg =(unsigned char*)Rg ->imageData+h*rgb->widthStep;  
//
//		for (int w=0;w<rgb->width;w++)   
//		{  
//
//			int s=pRGB[R]+pRGB[G]+pRGB[B];  
//
//			if (s< 120)
//			{
//				prg [R]=0;
//				prg [G]=0;
//				prg [B]=255;
//			} 
//
//			pRGB+=3;  
//			prg +=3;
//		}  
//	} 
//	return 1;
//}
//
//int testSkin(int argc, char* argv[])  
//{     
//	std::string Filepath;//(argv[1]);
//	//Filepath=("E:/Image/finger/finger3.jpg");
//	Filepath=("E:/Image/finger/fingerS06.png");
//	Filepath=("E:/Image/track/finger.jpg");
//	IplImage* img= cvLoadImage(Filepath.c_str() ); //随便放一张jpg图片在D盘或另行设置目录  
//
//	IplImage* dstRGB=cvCreateImage(cvGetSize(img),8,3);  
//	IplImage* dstRG=cvCreateImage(cvGetSize(img),8,1);  
//	IplImage* dstRG_NB=cvCreateImage(cvGetSize(img),8,3); 
//	IplImage* dst_crotsu=cvCreateImage(cvGetSize(img),8,1);  
//	IplImage* dst_YUV=cvCreateImage(cvGetSize(img),8,3);  
//	IplImage* dst_HSV=cvCreateImage(cvGetSize(img),8,3);  
//
//
//	//It must be a single channal image!
//	IplImage* ImgEq=cvCreateImage(cvGetSize(img),8,3); 
//	//cvEqualizeHist(img,ImgEq);
//
//	//cutBlack(ImgEq,dstRG_NB);
//	cutBlack(img,dstRG_NB);
//	cvShowImage("dstRG_NB", dstRG_NB);
//	//cvWaitKey(0);  
//
//	// cvNamedWindow("inputimage", CV_WINDOW_AUTOSIZE);  
//	cvShowImage("inputimage", img);  
//	//cvWaitKey(0);  
//
//	// SkinRGB(img,dstRGB);  
//	// //cvNamedWindow("dstRGB", CV_WINDOW_AUTOSIZE);  
//	// cvShowImage("dstRGB", dstRGB);  
//	//// cvWaitKey(0);  
//
//
//	//cvSkinRG(img,dstRG); 
//	cvSkinRG(dstRG_NB,dstRG);
//
//	//cvNamedWindow("dstRG", CV_WINDOW_AUTOSIZE);  
//	cvShowImage("dstRG", dstRG);    
//	// cvWaitKey(0);  
//	//   cvSkinOtsu(img,dst_crotsu);  
//	//   //cvNamedWindow("dst_crotsu", CV_WINDOW_AUTOSIZE);  
//	//cvShowImage("dst_crotsu", dst_crotsu);  
//	//  // cvWaitKey(0);  
//	//   cvSkinYUV(img,dst_YUV);  
//	//   cvNamedWindow("dst_YUV", CV_WINDOW_AUTOSIZE);  
//	//   cvShowImage("dst_YUV", dst_YUV);  
//	//  // cvWaitKey(0);  
//
//	//   cvSkinHSV(img,dst_HSV);  
//	//   //cvNamedWindow("dst_HSV", CV_WINDOW_AUTOSIZE);  
//	//   cvShowImage("dst_HSV", dst_HSV);  
//
//	cvWaitKey(0);  
//	return 0;  
//} 
//
//// skin region location using rgb limitation  
//void SkinRGB(IplImage* rgb,IplImage* _dst)  
//{  
//	assert(rgb->nChannels==3&& _dst->nChannels==3);  
//
//	static const int R=2;  
//	static const int G=1;  
//	static const int B=0;  
//
//	IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3); 
//
//
//	cvZero(dst);  
//
//	for (int h=0;h<rgb->height;h++) {  
//		unsigned char* prgb=(unsigned char*)rgb->imageData+h*rgb->widthStep;  
//		unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
//
//		for (int w=0;w<rgb->width;w++) {  
//
//
//			if ((prgb[R]>95 && prgb[G]>40 && prgb[B]>20 &&  
//				prgb[R]-prgb[B]>15 && prgb[R]-prgb[G]>15/*&& 
//														!(prgb[R]>170&&prgb[G]>170&&prgb[B]>170)*/)||//uniform illumination   
//														(prgb[R]>200 && prgb[G]>210 && prgb[B]>170 &&  
//														abs(prgb[R]-prgb[B])<=15 && prgb[R]>prgb[B]&& prgb[G]>prgb[B])//lateral illumination  
//														) {  
//															memcpy(pdst,prgb,3);  
//			}             
//			prgb+=3;  
//			pdst+=3; 
//
//		}  
//	}  
//	cvCopyImage(dst,_dst); 
//
//	cvReleaseImage(&dst);  
//}  
//// skin detection in rg space  
//void cvSkinRG(IplImage* rgb,IplImage* gray)  
//{  
//	assert(rgb->nChannels==3&&gray->nChannels==1);  
//
//	const int R=2;  
//	const int G=1;  
//	const int B=0;  
//
//
//	IplImage* Rg =cvCreateImage(cvGetSize(rgb),8,3); 
//	//IplImage* Rg=NULL;
//	cvCopyImage(rgb,Rg);
//
//	double Aup=-1.8423;  
//	double Bup=1.5294;  
//	double Cup=0.0422;  
//	double Adown=-0.7279;  
//	double Bdown=0.6066;  
//	double Cdown=0.1766;  
//	for (int h=0;h<rgb->height;h++) {  
//		unsigned char* pGray=(unsigned char*)gray->imageData+h*gray->widthStep;  
//		unsigned char* pRGB=(unsigned char* )rgb->imageData+h*rgb->widthStep;  
//		unsigned char* prg =(unsigned char*)Rg ->imageData+h*rgb->widthStep;  
//
//		for (int w=0;w<rgb->width;w++)   
//		{  
//
//			int s=pRGB[R]+pRGB[G]+pRGB[B];  
//			int iRG = pRGB[R]+pRGB[G];
//			double r=(double)pRGB[R]/s;  
//			double g=(double)pRGB[G]/s;  
//			double Gup=Aup*r*r+Bup*r+Cup;  
//			double Gdown=Adown*r*r+Bdown*r+Cdown;  
//			double Wr=(r-0.33)*(r-0.33)+(g-0.33)*(g-0.33);  
//
//
//			//得到RG图像
//			prg [B]=0;
//			//prg [G]=0;
//			prg [R]=Wr*256;
//
//			//指甲处的G值会高一些！比较亮！
//			if (pRGB[B] >50 || s >600 || iRG <50 || g<Gup && g>Gdown && Wr>0.004 || pRGB[R]-pRGB[B]<70 )  
//			{  
//				*pGray=255;  
//			}  
//			else  
//			{   
//				*pGray=0;  
//			}  
//
//			pGray++;  
//			pRGB+=3;  
//
//			prg +=3;
//		}  
//	}  
//
//	//对图像进行腐蚀膨胀、骨架细化
//	//默认使用3*3的模板
//	IplImage* RgSkeleton =cvCreateImage(cvGetSize(gray),8,1); 
//	cvCopyImage(gray,RgSkeleton);
//
//	//cvDilate(RgSkeleton,RgSkeleton,NULL,5);
//	
//	cvShowImage("RgSkeletonD", RgSkeleton);
//	
//	//cvErode(RgSkeleton,RgSkeleton,NULL,5);
//	cvShowImage("RgSkeletonR", RgSkeleton);
//
//	//cvSaveImage("E:/Image/finger/fingerRg.jpg",Rg);
//	//cvShowImage("dst_crImage", Rg);
//	cvReleaseImage(&Rg);
//	//cvWaitKey(0);
//
//}  
// implementation of otsu algorithm  
// author: onezeros#yahoo.cn  
// reference: Rafael C. Gonzalez. Digital Image Processing Using MATLAB  
void cvThresholdOtsu(IplImage* src, IplImage* dst)  
{  
	int height=src->height;  
	int width=src->width;  

	//histogram  
	float histogram[256]={0};  
	for(int i=0;i<height;i++) {  
		unsigned char* p=(unsigned char*)src->imageData+src->widthStep*i;  
		for(int j=0;j<width;j++) {  
			histogram[*p++]++;  
		}  
	}  
	//normalize histogram  
	int size=height*width;  
	for(int i=0;i<256;i++) {  
		histogram[i]=histogram[i]/size;  
	}  

	//average pixel value  
	float avgValue=0;  
	for(int i=0;i<256;i++) {  
		avgValue+=i*histogram[i];  
	}  

	int threshold;    
	float maxVariance=0;  
	float w=0,u=0;  
	for(int i=0;i<256;i++) {  
		w+=histogram[i];  
		u+=i*histogram[i];  

		float t=avgValue*w-u;  
		float variance=t*t/(w*(1-w));  
		if(variance>maxVariance) {  
			maxVariance=variance;  
			threshold=i;  
		}  
	}  

	cvThreshold(src,dst,threshold,255,CV_THRESH_BINARY);  
}  

//void cvSkinOtsu(IplImage* src, IplImage* dst)  
//{  
//	assert(dst->nChannels==1&& src->nChannels==3);  
//
//	IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);  
//	IplImage* cr=cvCreateImage(cvGetSize(src),8,1);  
//	cvCvtColor(src,ycrcb,CV_BGR2YCrCb);  
//	cvSplit(ycrcb,0,cr,0,0);  
//
//	cvThresholdOtsu(cr,cr);  
//	cvCopyImage(cr,dst);  
//	cvReleaseImage(&cr);  
//	cvReleaseImage(&ycrcb);  
//}  
//
//void cvSkinYUV(IplImage* src,IplImage* dst)  
//{  
//	IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);  
//	//IplImage* cr=cvCreateImage(cvGetSize(src),8,1);  
//	//IplImage* cb=cvCreateImage(cvGetSize(src),8,1);  
//	cvCvtColor(src,ycrcb,CV_BGR2YCrCb);  
//	//cvSplit(ycrcb,0,cr,cb,0);  
//
//	static const int Cb=2;  
//	static const int Cr=1;  
//	static const int Y=0;  
//
//	//IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3);  
//	cvZero(dst);  
//
//	for (int h=0;h<src->height;h++) {  
//		unsigned char* pycrcb=(unsigned char*)ycrcb->imageData+h*ycrcb->widthStep;  
//		unsigned char* psrc=(unsigned char*)src->imageData+h*src->widthStep;  
//		unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
//		for (int w=0;w<src->width;w++) {  
//			if (pycrcb[Cr]>=133&&pycrcb[Cr]<=173&&pycrcb[Cb]>=77&&pycrcb[Cb]<=127)  
//			{  
//				memcpy(pdst,psrc,3);  
//			}  
//			pycrcb+=3;  
//			psrc+=3;  
//			pdst+=3;  
//		}  
//	}  
//	//cvCopyImage(dst,_dst);  
//	//cvReleaseImage(&dst);  
//}  
//
//void cvSkinHSV(IplImage* src,IplImage* dst)  
//{  
//	IplImage* hsv=cvCreateImage(cvGetSize(src),8,3);  
//	//IplImage* cr=cvCreateImage(cvGetSize(src),8,1);  
//	//IplImage* cb=cvCreateImage(cvGetSize(src),8,1);  
//	cvCvtColor(src,hsv,CV_BGR2HSV);  
//	//cvSplit(ycrcb,0,cr,cb,0);  
//
//	static const int V=2;  
//	static const int S=1;  
//	static const int H=0;  
//
//	//IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3);  
//	cvZero(dst);  
//
//	for (int h=0;h<src->height;h++) {  
//		unsigned char* phsv=(unsigned char*)hsv->imageData+h*hsv->widthStep;  
//		unsigned char* psrc=(unsigned char*)src->imageData+h*src->widthStep;  
//		unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
//		for (int w=0;w<src->width;w++) {  
//			if (phsv[H]>=7&&phsv[H]<=29)  
//			{  
//				memcpy(pdst,psrc,3);  
//			}  
//			phsv+=3;  
//			psrc+=3;  
//			pdst+=3;  
//		}  
//	}  
//	//cvCopyImage(dst,_dst);  
//	//cvReleaseImage(&dst);  
//}  

int  testKP(int argc, char* argv[])
{
	//char* Path = argv[1];
	std::string Filepath(argv[2] );
	//Filepath=("E:/Image/finger/finger3.jpg");
	//Filepath=("E:/Image/finger/fingerRgB.png");
	//Filepath=("E:/DataBase/Eye/1072-04-15/474.727.A.20160219141406945.00.bmp");
	IplImage* img= cvLoadImage(Filepath.c_str() ); //随便放一张jpg图片在D盘或另行设置目录  

	IplImage* dstRGB=cvCreateImage(cvGetSize(img),8,3);  
	IplImage* dstRG=cvCreateImage(cvGetSize(img),8,1);  
	IplImage* dst_crotsu=cvCreateImage(cvGetSize(img),8,1);  
	IplImage* dst_YUV=cvCreateImage(cvGetSize(img),8,3);  
	IplImage* dst_HSV=cvCreateImage(cvGetSize(img),8,3);  


	//IplImage* src=img;
	//IplImage* Keypoints=nullptr;
	

	IplImage*  Edge =cvCreateImage(cvGetSize(img),8,1);
	cvCannyImg(img,Edge);
	cvShowImage("Edge", Edge);  

	Mat Image =imread(Filepath.c_str(),1);
	Mat Keypoints =imread(Filepath.c_str(),1);
	cvFingerKeypoints(Image,Keypoints) ;

	cvNamedWindow("inputimage", CV_WINDOW_AUTOSIZE);  
	cvShowImage("inputimage",  img);  
	cvShowImage("dst_HSV", dst_HSV);  

	cvWaitKey(0);  
	return 0;  
}

void cvFingerKeypoints(cv::Mat image,cv::Mat  KeyPoints) 
{
	cv::Mat  Descriptors;
	vector<KeyPoint>  KPoints;KPoints.resize(0);
	initModule_nonfree();


	Ptr<Feature2D>  sift = Algorithm::create<Feature2D>("Feature2D.SIFT");
	(*sift)(image,noArray(), KPoints, Descriptors ) ;
	Ptr<FeatureDetector >  FastDetect= Algorithm::create<FeatureDetector>("Feature2D.FAST");
	Ptr<DescriptorExtractor >  FastExtract= Algorithm::create<DescriptorExtractor>("Feature2D.FAST");
	FastDetect->detect(image,KPoints);
	//FastExtract->compute(image,KPoints,Descriptors);//why 内存错误？？？

	//Scalar S(0,255,0);
	drawKeypoints(image,KPoints,image,Scalar(0,255,0),4);
	imshow("image",image);
	waitKey(0);
	return ;
}

void cvFingerKeypoints(IplImage* src,IplImage* Keypoints) 
{

	return ;
}

//
//int testfingerTrack(int argc, char* argv[])
//{
//	CvCapture* capture = cvCreateFileCapture("E:\\sample.avi" );
//	IplImage* frame=NULL;
//	IplImage* dstRG=cvCreateImage(cvGetSize(frame),8,1);
//
//	while (1){
//
//		frame = cvQueryFrame(capture);
//		if (!frame)   
//			break;
//
//		cvShowImage("avi" ,frame);
//		cvOutlineFinger(frame,dstRG);
//
//		char c = cvWaitKey(240);
//		if (c == 27)   break ;
//	}
//
//	cvReleaseCapture(&capture); 
//	return 1;
//}
//
//int cvOutlineFinger(IplImage* rgb,IplImage* gray)
//{
//	assert(rgb->nChannels==3&&gray->nChannels==1);  
//
//	const int R=2;  
//	const int G=1;  
//	const int B=0;  
//
//	double Aup=-1.8423;  
//	double Bup=1.5294;  
//	double Cup=0.0422;  
//	double Adown=-0.7279;  
//	double Bdown=0.6066;  
//	double Cdown=0.1766;  
//
//	for (int h=0;h<rgb->height;h++) {  
//		unsigned char* pGray=(unsigned char*)gray->imageData+h*gray->widthStep;  
//		unsigned char* pRGB=(unsigned char* )rgb->imageData+h*rgb->widthStep;  
//
//		for (int w=0;w<rgb->width;w++)   
//		{  
//			int s=pRGB[R]+pRGB[G]+pRGB[B];  
//			int iRG = pRGB[R]+pRGB[G];
//			double r=(double)pRGB[R]/s;  
//			double g=(double)pRGB[G]/s;  
//			double Gup=Aup*r*r+Bup*r+Cup;  
//			double Gdown=Adown*r*r+Bdown*r+Cdown;  
//			double Wr=(r-0.33)*(r-0.33)+(g-0.33)*(g-0.33);  
//
//			//处理为蓝色背景
//			if (s< 120)
//			{
//				pRGB[R]=0;
//				pRGB[G]=0;
//				pRGB[B]=255;
//			} 
//
//			//指甲处的G值会高一些！比较亮！
//			if (pRGB[B] >50 || s >600 || iRG <50 || g<Gup && g>Gdown && Wr>0.004 || pRGB[R]-pRGB[B]<80 )  
//			{  
//				*pGray=255;  
//			}  
//			else  {   
//				*pGray=0;  
//			}  
//
//			pGray++;  
//			pRGB+=3;  
//		}  
//	}  
//
//	//对图像进行腐蚀膨胀、骨架细化
//	//默认使用3*3的模板
//	cvDilate(gray,gray,NULL,5);
//	cvErode(gray,gray,NULL,5);
//
//	return 1;
//}
//
//int testTrack(int argc, char* argv[])
//{
//	//CvCapture   Cap;
//	//("E:/Image/track/track.avi");
//	//CvCapture* capture = cvCreateFileCapture("E:/Image/track/track.avi" );
//	CvCapture* capture = cvCreateFileCapture("D:/CodeBase/IisuVR/motionTracking/test/camera.avi");
//
//	IplImage* frame = NULL;
//	//IplImage* dstRG=cvCreateImage(cvGetSize(frame),8,1);
//
//	int Idx =0;
//	while (1){
//
//		frame = cvQueryFrame(capture);
//		cv::Mat fM = frame;
//		CvSize size;
//		size.height =(frame->height);
//		size.width  =(frame->width);
//		IplImage *GrayI= cvCreateImage( size,8,1);
//		cvCvtColor(frame,GrayI,CV_BGR2GRAY);
//		cvCanny(GrayI,GrayI,30,120,3);
//		cvNot(GrayI,GrayI);
//
//		if (!frame)   
//			break;
//		cv::Mat GrayC(fM.rows,fM.cols,CV_8UC3);
//		GrayC =fM.clone();
//		cv::Mat Gray(fM.rows,fM.cols,CV_8UC1);
//
//		cv::cvtColor(GrayC,Gray,CV_BGR2GRAY);
//		cv::Mat Bin = Gray.clone();
//
//		cv::Canny(Gray,Bin,30,120,3,false);
//		
//		//cv::adaptiveThreshold(Gray,Bin,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,3,5);
//
//		cvShowImage("avi" ,frame);
//		cvShowImage("canny" ,GrayI);
//		imshow("Binary",Bin);
//		//cvOutlineFinger(frame,dstRG);
//		//IplImage* Edge=cvCreateImage(cvGetSize(frame),8,1);
//		//cvCanny(frame,Edge,50,150,3);
//		//cvShowImage("Canny" ,Edge);
//
//		//cvSkinRG(frame,Edge);
//		
//		//char c = cvWaitKey(240);
//		//if (c == 27)   break ;
//		cvWaitKey(20);
//		std::string PathMe = "D:/CodeBase/IisuVR/motionTracking/test/Picture/";
//		std::strstream  ss;
//		ss.clear();
//		//Idx>>ss;
//		ss<<Idx;
//
//		std::string SC;
//		//SC<<ss;
//		ss>>SC;
//
//		PathMe.append(SC);
//		PathMe.append(".jpg");
//		//cvSaveImage(PathMe.c_str(),frame);
//		//cvSaveImage("E:/Image/finger/fingerRg.jpg",Rg);
//
//		++Idx;
//		//cvReleaseImage(&Edge);
//	}
//
//	//cvReleaseImage(&Edge);
//	cvReleaseCapture(&capture); 
//	return 1;
//}

void cvCannyImg(IplImage* rgb, IplImage*  Edge)
{
	cvCanny(rgb,Edge,30,120,3);
	return ;
}

//
//void testFast(int argc, char* argv[])
//{
//	cv::Mat        im;
//	double  threshold = 20;//调小阈值会得到更多的检测结果！参数 80、40、20.
//	std::vector<CvPoint>    coords(0);
//
//	//CvCapture* capture = cvCreateFileCapture("C:/Users/wishchin/Desktop/video/Slam/Slam2.mp4");
//	CvCapture* capture = cvCreateFileCapture("C:/Users/wishchin/Desktop/picture/slam/NoAutoExposure.avi");
//
//	IplImage* frame = NULL;
//	//IplImage* dstRG=cvCreateImage(cvGetSize(frame),8,1);
//
//	//快速角点检测  
//	std::vector<cv::KeyPoint>  keypoints;  
//
//	int Idx =0;
//	while (1){
//
//		frame = cvQueryFrame(capture);
//		cv::Mat im = frame;
//
//		cv::Mat  image(im.rows, im.cols, 1);
//		if (3== im.channels() ){
//			cv::cvtColor (im,image,CV_BGR2GRAY);  
//		}
//		else{
//			//cvCopyImage(im,image);
//			image =im.clone();
//		}
//
//		cv::FastFeatureDetector fast(threshold , false);  
//
//		SYSTEMTIME sys; 
//		GetLocalTime( &sys ); 
//		printf( "%4d/%02d/%02d %02d:%02d:%02d.%03d 星期%1d  帧数%1d\n",sys.wYear,sys.wMonth,sys.wDay,sys.wHour,sys.wMinute, sys.wSecond,sys.wMilliseconds,sys.wDayOfWeek, Idx); 
//		//int MileT = sys.wMilliseconds;
//
//		fast .detect (image,keypoints);  
//
//		GetLocalTime( &sys ); 
//		printf( "%4d/%02d/%02d %02d:%02d:%02d.%03d 星期%1d  帧数%1d\n",sys.wYear,sys.wMonth,sys.wDay,sys.wHour,sys.wMinute, sys.wSecond,sys.wMilliseconds,sys.wDayOfWeek, Idx); 
//		//int MileT = sys.wMilliseconds;
//
//		//cv::FAST(image,keypoints,threshold,false);//cv::FAST(image,keypoints,threshold,true);//
//		cv::drawKeypoints (image ,keypoints,  image,cv::Scalar::all(255),  cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); 
//
//		//coords.resize(keypoints.size() );
//		cv::imshow("image",image);
//
//		cvWaitKey(100);
//		//std::string PathMe = "D:/CodeBase/IisuVR/motionTracking/test/Picture/";
//		//std::strstream  ss;
//		//ss.clear();
//		//ss<<Idx;
//		//std::string SC;
//		//ss>>SC;
//
//		//PathMe.append(SC);
//		//PathMe.append(".jpg");
//
//		++Idx;
//	}
//
//	//cvReleaseImage(&Edge);
//	cvReleaseCapture(&capture); 
//	return ;
//}

int testGlintRemover(int argc, char* argv[])
{	
	
	int FlagInput = 0;
	std::string PicsSeqFolder;
	//const char* Method = argv[1];
	if (argc>1){
		FlagInput = atoi(argv[1] );

		std::string PicsFolder(argv[2] );
		PicsSeqFolder = PicsFolder;
	}

	
	std::string   Extention(argv[3]);
	std::vector<std::string>  Filelist(0);

	if(FlagInput > 0){
		//getImageSeq( PicsSeqFolder, Extention, Filelist);//使用boost出现失误，不是正确排序的。
		if(Filelist.size() < 1 )
		{
			const boost::filesystem::path  base_dir(PicsSeqFolder );
			const std::string extension = Extention;
			loadFileList(base_dir, extension, Filelist);  
		}
	}

	std::string Filepath;//argv[3] );
	for (int i=0;i<Filelist.size();++i)
	{
		Filepath =Filelist[i];

		//1.测试Mat
		cv::Mat matSrc =imread(Filepath.c_str(),1);
		cv::Mat matDst =matSrc.clone();

		CGlintRemover  GlintRemoverM(matSrc);
		GlintRemoverM.PreProcess();
		GlintRemoverM.Process();
		GlintRemoverM.GetOutcome(matDst);

		cv::imshow("matSrc",matSrc);
		cv::imshow("matDst",matDst);

		std::string FilepathDst=Filepath;
		FilepathDst.append("arc.bmp");
		cv::imwrite(FilepathDst.c_str(),matDst);

		cv::waitKey(1000);

		////1.测试IplImage
		//IplImage*  imageSrc = cvLoadImage(Filepath.c_str() ); //随便放一张jpg图片在D盘或另行设置目录  
		//IplImage*  imageDst = cvCreateImage(cvGetSize(imageSrc ),8,1);

		//CGlintRemover  GlintRemoverI(imageSrc);
		//GlintRemoverI.Process();
		//GlintRemoverI.GetOutcome(imageDst);

		//cvShowImage("imageSrc", imageSrc);
		//cvShowImage("imageDst", imageDst);  
		//cvWaitKey(500); 
	}

	return 1;
}

int testRectSelector(int argc, char* argv[])
{
	std::vector<cv::Mat> patchList(0);

	//1.测试Mat
	std::string Filepath(argv[2] );
	cv::Mat matSrc =imread(Filepath.c_str(),1);

	//CRectSelect Selector(matSrc.cols,matSrc.rows);
	CRectSelect Selector(matSrc );
	Selector.GetPicPatches(patchList);

	const std::string FlagC = "Flag"; 
	std::strstream   SFlagC; 
	std::string  SNum;
	for (int i=0;i< patchList.size();++i )
	{
		SFlagC.clear();
		SNum.clear();

		SFlagC<< i;
		SFlagC>> SNum;

		std::string Flag = FlagC;
		Flag.append(SNum);

		cv::imshow(Flag.c_str(),patchList[i]);
	}

	cv::waitKey(0);
	return 1;
}

bool loadFileList(const boost::filesystem::path &base_dir, const std::string &extension,  
				  std::vector<std::string> &FileList)  
{  
	if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))  
		return true;  

	boost::filesystem::directory_iterator it(base_dir);  

	for (;  
		it != boost::filesystem::directory_iterator ();  
		++it)  
	{  
		if (boost::filesystem::is_directory (it->status ()))  
		{  
			std::stringstream ss;  
			ss << it->path ();  
			loadFileList (it->path (), extension, FileList);  
		}  
		if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == extension)  
		{  
			std::string Path;  
			Path =base_dir.string();  
			Path.append("/");  
			Path.append(it->path().filename().string());    
			FileList.push_back (Path);  
		}  
	}  
	return (true);  
}  