#include "stdafx.h"
#include "GlintRemover.h"


CGlintRemover::CGlintRemover(void)
{
}


CGlintRemover::CGlintRemover(IplImage*  imageSrc)
{
	this->mImageOut = cvCreateImage(cvGetSize(imageSrc ),8,1);
}

CGlintRemover::CGlintRemover(cv::Mat   &src)
{
	this->mMatOut = src.clone();
	this->mImageHeight = this->mMatOut.rows;
	this->mImageWidth  = this->mMatOut.cols;
}


CGlintRemover::~CGlintRemover(void)
{
}

int CGlintRemover::PreProcess()
{
	//ȥ����Χ�ĺ�Ȧ��ʹ��houghԲ��⣬ȥ������Բ
	//ʹ����ֵ120�������
	this->RemoveBackGround();

	return 1;
}

int CGlintRemover::Process()
{
	this->DetectGlint();

	this->RemoveGlint();


	return 1;
}

int CGlintRemover::GetOutcome(IplImage*  dst)
{

	return 1;
}

int CGlintRemover::GetOutcome(cv::Mat   &dst)
{
	this->mMatOut.copyTo(dst);
	return 1;
}

//ȥ����Χ�ĺ�Ȧ��ʹ��houghԲ��⣬ȥ������Բ
//ʹ����ֵ120�������
int CGlintRemover::RemoveBackGround()
{
	
	cv::Mat  dst = this->circleCheckHough(this->mMatOut);
	cv::imshow("dst",dst);
	cv::waitKey(0);

	return 1;
}

int CGlintRemover::DetectGlint()
{

	this->SearchBrightnessPeak();

	this->SearchThresholdLowerBound();

	return 1;
}

//ȥ����㣬��С��ѡ�񣬸��ݴ�С�ж��Ƿ�ȥ��
int CGlintRemover::RemoveGlint()
{



	return 1;
}

double CGlintRemover::SearchBrightnessPeak()
{

	this->mMatOut;
	//����һ����ͨ��ͼ��  
	cv::Mat image(this->mImageHeight ,this->mImageWidth, CV_8U, 1);
	cv::cvtColor(this->mMatOut,image,CV_BGR2GRAY);
	cv::Mat histogram;  //cv::MatND histogram;  

	//256������Χ��0��255.  
	//nbsp;
	//const 
	int histSize = 256;  

	//std::vector<int > histSize(256);
	float range[] = {0, 255};  
	const float *ranges = { range };  
	const int channels = 0;  

	//cv::calcHist(image, 1, &channels, cv::Mat(), histogram, 1, &histSize, &ranges[0], true, false); 
	//cv::calcHist(image, 1, channels, cv::Mat(), histogram, 1, histSize, ranges[0], true, false); 
	cv::calcHist(&image, 1, 0,cv::Mat(), histogram,1, &histSize, &ranges, true, false); 
	int row = histogram.rows;  
	int col = histogram.cols;  

	float *h = (float*)histogram.data;  
	double hh[256];  
	if (h) {  
		for (int i = 0; i < 256; ++i) {  
			hh[i] = h[i];  
			std::cout<< hh[i] << std::endl;
		}  
	} 

	this->DrawHist(hh);

	return 1;
}

double CGlintRemover::SearchThresholdLowerBound()
{

	return 1;
}


double CGlintRemover::DrawHist(double hh[256])
{
	cv::Mat canvas( this->mMatOut.rows ,256*4, CV_8U, 1);

	for(int i=0;i< 256; ++i){
		//for(int j=hh[i];j< this->mMatOut.rows;++i){
		//}
		int Pos = ( (int)(hh[i]) / 10 ) %( this->mMatOut.rows);
		canvas.at<uchar>( this->mMatOut.rows - Pos-1,i*4 ) =255;
	}

	cv::imshow("canvas",canvas);
	cv::waitKey(0);

	return 1;
}

cv::Mat  CGlintRemover::circleCheckHough(cv::Mat  &Src)
{
	IplImage   GrayImage = Src;
	//IplImage* pGrayImage =NULL;
	IplImage *pGrayImage =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);    
	//pGrayImage =&GrayImage;
	if (GrayImage.nChannels >1)
	{
		cvCvtColor(&GrayImage, pGrayImage, CV_BGR2GRAY); 
	}
	else
	{
		pGrayImage= &GrayImage; 
	}
	  

	CvMemStorage *pcvMStorage = cvCreateMemStorage();    
	double fMinCircleGap = pGrayImage->height / 10.0;    
	//CvSeq *pcvSeqCircles = cvHoughCircles(pGrayImage, pcvMStorage, CV_HOUGH_GRADIENT, 1, fMinCircleGap);
	CvSeq *pcvSeqCircles = cvHoughCircles(pGrayImage, pcvMStorage, CV_HOUGH_GRADIENT, 1, fMinCircleGap);  

	// ����ֱ��    
	IplImage *pColorImage = cvCreateImage(cvGetSize(pGrayImage), IPL_DEPTH_8U, 3);    
	cvCvtColor(pGrayImage, pColorImage, CV_GRAY2BGR);  

	int i =0;
	for (i = 0; i < pcvSeqCircles->total; i++)  {    
		float* p = (float*)cvGetSeqElem(pcvSeqCircles, i);    
		cvCircle(pColorImage, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(255, 0, 0), 5);    
	}    

	cv::Mat  Dst(&(*pColorImage));
	return Dst;
}