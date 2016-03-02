#include "stdafx.h"
#include "GlintRemover.h"

using namespace std;
using namespace  cv;

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

	this->mThresholdInvLowerBound  = -1; //�������ֵ���½�
	this->mThresholdInvHigherBound = 16;//�������ֵ���Ͻ�

	this->mThresholdPupilLowerBound    =  -1;  //�������ֵ���½�
	this->mThresholdPupilHigherBound   =  25; 
	this->mThresholdPupilInvLowerBound = 230;
	this->mThresholdPupilInvHigherBound= 255; //�������ֵ�ķ�ת�½�

	//��⻡�� ������С--Բ�ܵ����½�
	this->mThresholdEdgePointsLowerBound =  60;
	this->mThresholdEdgePointsHigherBound=  400;
}


CGlintRemover::~CGlintRemover(void)
{
	cvReleaseImage(&this->mImageOut);
}

//�õ���һЩ�����Եı�Ե
int CGlintRemover::PreProcess()
{
	//ȥ����Χ�ĺ�Ȧ��ʹ��houghԲ��⣬ȥ������Բ
	//ʹ����ֵ120�������
	//this->RemoveBackGround();

	cv::Mat  ImageColor;
	ImageColor = this->mMatOut.clone();
	//ImageColor = this->CircleCheckHough(this->mMatOut);;

	//this->DetectArc (ImageColor);
	this->DetectArcRs (ImageColor);
	this->mMatOut =ImageColor.clone();


	return 1;
}

int CGlintRemover::Process()
{
	//this->DetectGlint();

	//this->RemoveGlint();


	return 1;
}

int CGlintRemover::GetOutcome(IplImage*  dst)
{

	return 1;
}

int CGlintRemover::GetOutcome(cv::Mat   &dst)
{
	//this->mMatOut.copyTo(dst);
	if (this->mMatOut.channels()>1 )
	{
		cv::cvtColor(this->mMatOut,dst,CV_BGR2GRAY);
	}
	return 1;
}

//ȥ����Χ�ĺ�Ȧ��ʹ��houghԲ��⣬ȥ������Բ
//ʹ����ֵ120�������
//�޷����Բ��ʹ�� �̶�ֵȥ���ڿ�
int CGlintRemover::RemoveBackGround()
{

	////ʹ��HoughԲ�����в���һ��
	//cv::Mat  dst = this->CircleCheckHough(this->mMatOut);

	CvPoint  center;
	center.x =320;
	center.y =240;
	int radius=200;
	//ʹ��ֱ�ӿ۳��ķ���ȥ����ɫ�߿�
	cv::Mat  dst;
	//ʹ�÷�ɫ����ȥ�����߼���ɫ�������
	//��ʹ����ֵ�˲���ȥ����Ե
	//Ч�����ǲ����
	//dst = this->CutBlackBlock(this->mMatOut,center, radius);


	////ʹ��HoughԲ�����в���һ��
	cv::Mat  dst2;
	dst2 = this->CircleCheckHough(this->mMatOut);

	//ʹ�ñ�Ե�����в���
	//dst = this->EdgeDetectBlock(this->mMatOut,center, radius);

	cv::imshow("dst",dst2);
	//cv::waitKey(0);
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

cv::Mat  CGlintRemover::CircleCheckHough(cv::Mat  &Src)
{
	//0.��ʹ��˫���˲��������˲���������Ե
	cv::Mat Src2 ;
	//cv::bilateralFilter( Src,Src2, 25, 25*2, 25/2 );
	cv::bilateralFilter( Src,Src2, 35, 35*2, 35/2 );
	//cv::imshow("Src2",Src2);

	IplImage   GrayImage = Src2;

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

	//1.�ȷ�ɫ
	for (int h=0;h<pGrayImage->height;h++) 
	{  
		unsigned char* pGray =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  pRGB =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  prg  =(unsigned char*)pGrayImage ->imageData+h*pGrayImage->widthStep;  

		for (int w=0;w<pGrayImage->width;w++)   
		{  
			*pGray =255- *pGray;  
			pGray++;  
		}
	}

	IplImage *pCanny =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);    
	cvCanny(pGrayImage, pCanny, 30, 220,3 );//���˫���˲���ʹ��

	cvShowImage( "pGrayImage ", pGrayImage );  
	cvShowImage( "pCanny",pCanny );
	//cvWaitKey(0);

	IplImage *pColorImage = cvCreateImage(cvGetSize(pGrayImage), IPL_DEPTH_8U, 3);    
	cvCvtColor(pGrayImage, pColorImage, CV_GRAY2BGR); 

	////1. ����ʹ��houghԲ��⣬����ʹ��Ѱ�ұ�Ե��findarc����
	//CvMemStorage* storage = cvCreateMemStorage (0);  
	//CvSeq* pcvSeqCircles = cvHoughCircles (pGrayImage, storage, CV_HOUGH_GRADIENT, 2, pGrayImage->width /15, 300, 100, 0, 200);   

	//// ���� ����  
	//for (int i = 0; i < pcvSeqCircles->total; i++)  {    
	//	float* p = (float*)cvGetSeqElem(pcvSeqCircles, i);    
	//	cvCircle(pColorImage, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(255, 0, 0), 5);    
	//}    

	//2. ����ʹ��Ѱ�ұ�Ե����ʹ��findarc����
	IplImage* dst = cvCreateImage( cvGetSize(pGrayImage), 8, 3 );    
	CvMemStorage* storage = cvCreateMemStorage(0);    
	CvSeq* contour = 0;    
	cvThreshold( pGrayImage, pGrayImage,225, 255, CV_THRESH_BINARY );//��ֵ��     
	cvNamedWindow( "Source", 1 );    
	cvShowImage( "Source", pGrayImage );  

	//��ȡ����     
	//CV_RETR_CCOMP���������е�����������������֯Ϊ���㣺�����Ǹ����ֵ��ⲿ�߽磬�ڶ����ǿն��ı߽磻
	cvFindContours( pGrayImage, 
		storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );    
	//storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );  
	////storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_CODE ); 
	//storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_CODE); 	

	cvZero( dst );//�������     
	CvSeq* _contour =contour;     
	double maxarea=0;    
	double minarea=1000;    
	std::vector<double> areaList(0);

	int n=-1,m=0;//nΪ����������������mΪ��������
	//�����ΪѰ����������
	for( ; contour != 0; contour = contour->h_next )    
	{    
		double tmparea=fabs(cvContourArea(contour));    
		if(tmparea < minarea)     
		{    
			cvSeqRemove(contour,0); //ɾ�����С���趨ֵ������     
			continue;    
		}    
		CvRect aRect = cvBoundingRect( contour, 0 );     
		//if ((aRect.width/aRect.height)<1)    
		//{    
		//	cvSeqRemove(contour,0); //ɾ����߱���С���趨ֵ������     
		//	continue;    
		//}    

		areaList.push_back(tmparea );
		if(tmparea > maxarea)    
		{    
			maxarea = tmparea;    
			n=m;    
		}    
		m++;    

		//max_level �������������ȼ�������ȼ�Ϊ0�����Ƶ��������������Ϊ1��������������������ͬ�ļ�����������     
		//���ֵΪ2�����е�����������ȼ�Ϊ2����������ͬ�����������е�һ��������������֡�     
		//���ֵΪ����������������ͬ�������������������ֱ������Ϊabs(max_level)-1����������
		CvScalar color = CV_RGB( 0, 255,255 ); //����һ��ɫ��ֵ 
		cvDrawContours( dst, contour, color, color, -1, 1, 8 );//�����ⲿ���ڲ������� 
		//if (m>0)  break;//һ����Ե����������Ȧ����ô�߳�������
	}    

	//Ѱ���������
	contour =_contour; /*int k=0;*/    
	int count= areaList.size();    
	//for( ; contour != 0; contour = contour->h_next )    
	//{    
	//	count++;    
	//	double tmparea=fabs(cvContourArea(contour));    
	//	if (tmparea==maxarea /*k==n*/)    
	//	{    
	//		CvScalar color = CV_RGB( 255, 0, 0);    
	//		cvDrawContours( dst, contour, color, color, -1, 1, 8 ); //�����ⲿ������      
	//	}    
	//	/*k++;*/    
	//}    

	count =m;
	printf("The total number of contours is:%d",count);     
	cvShowImage( "Components", dst );    
	//cvWaitKey(0);    

	//cv::Mat  Dst(&(*pColorImage));
	cv::Mat  Dst(&(*dst));

	cvReleaseImage(&pCanny); 
	cvReleaseImage(&pGrayImage);  
	//cvReleaseImage(&dst);     

	return Dst;
}

//ʹ�ù̶�ֵȥ���ڱ�//���ز�ɫͼ��
cv::Mat  CGlintRemover::CutBlackBlock(cv::Mat  &Src,CvPoint  &Center, int Radius)
{
	//0.��ʹ��˫���˲��������˲���������Ե
	cv::Mat Src2 ;
	cv::bilateralFilter( Src,Src2, 25, 25*2, 25/2 );
	cv::imshow("Src2",Src2);


	IplImage   GrayImage = Src2;

	IplImage *pGrayImage =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);    

	if (GrayImage.nChannels >1)
	{
		cvCvtColor(&GrayImage, pGrayImage, CV_BGR2GRAY); 
	}
	else
	{
		pGrayImage= &GrayImage; 
	}


	for (int h=0;h<pGrayImage->height;h++) {  
		unsigned char* pGray =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  pRGB =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  prg  =(unsigned char*)pGrayImage ->imageData+h*pGrayImage->widthStep;  

		for (int w=0;w<pGrayImage->width;w++)   
		{  
			//1.�ȷ�ɫ
			*pGray =255- *pGray;  
			pGray++;  
		}
	}

	IplImage *pCanny =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);    
	cvCanny(pGrayImage, pCanny, 30, 220, 3 );//���˫���˲���ʹ��

	cvShowImage( "contour", pGrayImage );  
	cvShowImage( "pCanny",pCanny );
	cvWaitKey(0);

	//2.ȥ������
	//2.1ʹ����ͨ�򷽷�����
	int invLowerBound  = this->mThresholdInvLowerBound ; //�������ֵ���½�
	int invHigherBound = this->mThresholdInvHigherBound;//�������ֵ���Ͻ�

	//2.1ʹ�ü򵥷���

	for (int h=0;h<pGrayImage->height;h++) {  
		unsigned char* pGray =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;   
		for (int w=0;w<pGrayImage->width;w++)   
		{  
			//2.1 �컯Ϊͫ��
			if(*pGray > invLowerBound &&*pGray < invHigherBound) 
			{
				//*pGray = this->mThresholdPupilInvLowerBound ;
				*pGray = mThresholdPupilInvHigherBound ;//Ӧ��ʹ���Ͻ�Ч�����ã�
			}
			pGray++;  
		}
	}

	//2.2 ʹ����̬ѧ����  
	IplImage *imgDilate = cvCreateImage(cvGetSize(pGrayImage), 8, 1);  

	//cvErode(  pGrayImage,imgDilate, NULL,5); //��ʴ  
	//cvDilate( pGrayImage,imgDilate, NULL,3); //����  

	//2.3 ʹ����ͨ�����
	IplImage* pContourImg = NULL;  
	CvMemStorage* storage = cvCreateMemStorage(0);  
	CvSeq* contour = 0;  
	CvSeq* contmax = 0;  
	int mode = CV_RETR_EXTERNAL; 

	//cvShowImage( "src", pImg );  
	//Ϊ������ʾͼ������ռ�  
	//3ͨ��ͼ���Ա��ò�ɫ��ʾ  
	pContourImg = cvCreateImage(
		cvGetSize(pGrayImage),  IPL_DEPTH_8U, 3);  

	//copy source image and convert it to BGR image  
	cvCvtColor(pGrayImage, pContourImg, CV_GRAY2BGR);

	//��ֵ�������contour  
	cvThreshold( pGrayImage, pGrayImage,225, 255, CV_THRESH_BINARY );//��ֵ��  
	//��ʾͼ��  
	cvShowImage( "contour", pGrayImage );  
	cvWaitKey(0); 

	cvFindContours( pGrayImage , storage, &contour, sizeof(CvContour),  
		mode, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0)); 

	//����������     
	cvCvtColor(pGrayImage, pContourImg, CV_GRAY2BGR);

	cvDrawContours(pContourImg, contour,  
		CV_RGB(255,0,0), CV_RGB(255, 0, 0),  
		2, 2, 8, cvPoint(0,0));  

	//��ʾͼ��  
	cvShowImage( "contour", pContourImg );  
	cvWaitKey(0);  

	// ����ֱ��    
	IplImage *pColorImage = cvCreateImage(cvGetSize(pGrayImage), IPL_DEPTH_8U, 3);   
	//IplImage *pColorImage = cvCreateImage(cvGetSize(imgDilate), IPL_DEPTH_8U, 3); 
	//IplImage *pColorImage = cvCreateImage(cvGetSize(pContourImg), IPL_DEPTH_8U, 3);
	cvCvtColor(pGrayImage, pColorImage, CV_GRAY2BGR);  


	cv::Mat  Dst(&(*pColorImage));
	//2.2.2 ʹ����ֵ�˲�
	cv::medianBlur(Dst,Dst,5);

	cvReleaseImage(&pCanny); 
	cvReleaseImage(&pGrayImage);  
	cvReleaseImage(&pColorImage);     
	return Dst;
}

//ʹ�ñ�Ե�����в���
cv::Mat  CGlintRemover::EdgeDetectBlock(cv::Mat  &src,CvPoint  &center, int radius)
{
	IplImage   GrayImage = src;
	IplImage *pGrayImage =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);    

	if (GrayImage.nChannels >1)
	{
		cvCvtColor(&GrayImage, pGrayImage, CV_BGR2GRAY); 
	}
	else
	{
		pGrayImage= &GrayImage; 
	}


	for (int h=0;h<pGrayImage->height;h++) {  
		unsigned char* pGray =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  pRGB =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  prg  =(unsigned char*)pGrayImage ->imageData+h*pGrayImage->widthStep;  

		for (int w=0;w<pGrayImage->width;w++)   
		{  
			//�ȷ�ɫ
			*pGray =255- *pGray;  
			pGray++;  
		}
	}


	// ����ֱ��    
	IplImage *pColorImage = cvCreateImage(cvGetSize(pGrayImage), IPL_DEPTH_8U, 3);    
	cvCvtColor(pGrayImage, pColorImage, CV_GRAY2BGR);  


	cv::Mat  Dst(&(*pColorImage));
	return Dst;

}


//ʹ�ñ�Ե���������½�Լ��	this->mThresholdEdgePointsLowerBound =100;
//this->mThresholdEdgePointsHigherBound= 20000;
int CGlintRemover::DetectArcRs (cv::Mat &ImageColor)
{
	cv::Mat frame =ImageColor.clone();
	cv::bilateralFilter( ImageColor,frame, 25, 25*2, 25/2 );
	imshow("bilateralFilter" , frame);

	IplImage   GrayImage = frame;
	IplImage *pGrayImage =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);   
	IplImage *pBinImage =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);   
	//pGrayImage =&GrayImage;

	if (GrayImage.nChannels >1)//if (frame.channels() >1)//
	{
		cvCvtColor(&GrayImage, pGrayImage, CV_BGR2GRAY); 
		//cv::cvtColor(frame , fgray , CV_RGB2GRAY);
	}
	else
	{
		pGrayImage= &GrayImage; 
	}

	//1.�ȷ�ɫ
	for (int h=0;h<pGrayImage->height;h++) 
	{  
		unsigned char* pGray =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  pRGB =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  prg  =(unsigned char*)pGrayImage ->imageData+h*pGrayImage->widthStep;  

		for (int w=0;w<pGrayImage->width;w++)   
		{  
			*pGray =255- *pGray;  
			pGray++;  
		}
	}

	//2.ȥ������
	//2.1ʹ����ͨ�򷽷�����
	int invLowerBound  = this->mThresholdInvLowerBound ; //�������ֵ���½�
	int invHigherBound = this->mThresholdInvHigherBound;//�������ֵ���Ͻ�

	//2.1ʹ�ü򵥷���
	//ȥ����ɫ�ߵ�
	for (int h=0;h<pGrayImage->height;h++) {  
		unsigned char* pGray =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;   
		for (int w=0;w<pGrayImage->width;w++)   
		{  
			//2.1 �컯Ϊͫ��//ȥ����ɫ�ߵ�
			if(*pGray > invLowerBound &&*pGray < invHigherBound) 
			{
				//*pGray = this->mThresholdPupilInvLowerBound ;
				*pGray = mThresholdPupilInvHigherBound ;//Ӧ��ʹ���Ͻ�Ч�����ã�
			}
			pGray++;  
		}
	}

	cv::Mat GrayImageM(pGrayImage);
	cv::Mat  img_bin;
	cv::medianBlur(GrayImageM,GrayImageM,5);//2.2 ȥ���ߵ�� ��...
	cv::threshold(GrayImageM , img_bin , 225, 255, CV_THRESH_BINARY);

	//imshow("ImageColor!" , (cv::Mat)ImageColor );
	//imshow("biFilter!"   , (cv::Mat)frame      );
	//imshow("pGrayImage!" , (cv::Mat)GrayImageM);
	//imshow("BinaryOut!"  , (cv::Mat)img_bin    );

	//cv::waitKey(0);

	//if(1)
	{
		vector<vector<cv::Point> >  all_contours;
		vector<vector<cv::Point> >      contours;

		vector<Vec4i> hierarchy;
		//cv::findContours( img_bin, all_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		cv::findContours( img_bin, all_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		//vector<vector<Point>>convex_contour(all_contours.size() );
		//3.1 ����һ�Σ���ȥС�ı�Ե��
		int edgeValid =0;
		for(int i=0;i<all_contours.size();i++)
		{
			if (
				all_contours[i].size() > this->mThresholdEdgePointsLowerBound
				&& all_contours[i].size() < this->mThresholdEdgePointsHigherBound )	
			
			{
				++edgeValid;
			}
		}

		vector<vector<Point>>convex_contour(edgeValid );

		vector<Point> approx_poly;
		vector<int>  apdNum(all_contours.size() );
		vector<RotatedRect >rects;
		vector<double>sigmas(all_contours.size() );
		double mv = 99999; int uind = 0;

		//1.����͹��Ե
		int idx =0;
		for(int i=0;i<all_contours.size();i++)
		{
			if (
				all_contours[i].size() > this->mThresholdEdgePointsLowerBound
				&& all_contours[i].size() < this->mThresholdEdgePointsHigherBound )	
			{
				//convexHull(all_contours[i] , convex_contour[i] , false , true);
				convexHull(all_contours[i] , convex_contour[idx] , false , true);

				//double ts = this->mLineDetector.isCircle(convex_contour[i]);
				double ts = this->mLineDetector.isCircle(convex_contour[idx]);
				sigmas[i] = ts;

				//RotatedRect rr = minAreaRect(convex_contour[i]);
				RotatedRect rr = minAreaRect(convex_contour[idx]);
				rects.push_back(rr);
				//approxPolyDP(convex_contour[i] , approx_poly , 1 , true);
				approxPolyDP(convex_contour[idx] , approx_poly , 1 , true);

				apdNum[i] = approx_poly.size();
				if(mv > ts && apdNum[i] >= 10){
					mv = ts;
					uind = i;
				}
				++idx;
			} 
			else
			{}
		}

		for(int i=0;i<all_contours.size();i++)
		{
			if (all_contours[i].size() >10)
			{
				drawContours(frame , all_contours , i , Scalar(255,0,0));
			}
		}

		for(int i=0;i<convex_contour.size();i++)
		{
			//drawContours(frame , convex_contour , uInds[i] , Scalar(0,0,255));
			drawContours(frame , convex_contour , i , Scalar(0,0,255));
		}

		//imshow("Contour" , frame);
		//waitKey(0);
	}

	ImageColor =frame.clone();
	return 0;
}

int CGlintRemover::DetectArc (cv::Mat &ImageColor)
{
	cv::Mat frame =ImageColor.clone();
	cv::bilateralFilter( ImageColor,frame, 25, 25*2, 25/2 );
	
	IplImage   GrayImage = frame;
	IplImage *pGrayImage =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);   
	IplImage *pBinImage =  cvCreateImage(cvGetSize(&GrayImage), IPL_DEPTH_8U, 1);   
	//pGrayImage =&GrayImage;

	if (GrayImage.nChannels >1)//if (frame.channels() >1)//
	{
		cvCvtColor(&GrayImage, pGrayImage, CV_BGR2GRAY); 
		//cv::cvtColor(frame , fgray , CV_RGB2GRAY);
	}
	else
	{
		pGrayImage= &GrayImage; 
	}

	//1.�ȷ�ɫ
	for (int h=0;h<pGrayImage->height;h++) 
	{  
		unsigned char* pGray =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  pRGB =(unsigned char*)pGrayImage->imageData+h*pGrayImage->widthStep;  
		//unsigned char*  prg  =(unsigned char*)pGrayImage ->imageData+h*pGrayImage->widthStep;  

		for (int w=0;w<pGrayImage->width;w++)   
		{  
			*pGray =255- *pGray;  
			pGray++;  
		}
	}

	//cv::adaptiveThreshold(
	//	fgray , img_bin , 255 , cv::ADAPTIVE_THRESH_GAUSSIAN_C ,
	//	cv::THRESH_BINARY_INV  , thresh_size , thresh_size / 3);
	//cv::threshold(fgray , img_bin , THRESH_BINARY_INV + THRESH_OTSU , 255 , 50);
	cvThreshold( pGrayImage , pBinImage ,225, 255, CV_THRESH_BINARY );//��ֵ��
	cv::Mat  img_bin(pBinImage);
	//cv::threshold(fgray , img_bin  , 20 , 255, CV_THRESH_BINARY);

	imshow("ImageColor!" , (cv::Mat)ImageColor );
	imshow("biFilter!"   , (cv::Mat)frame      );
	imshow("pGrayImage!" , (cv::Mat)pGrayImage );
	imshow("BinaryOut!"  , (cv::Mat)img_bin    );
	cv::waitKey(0);

	//if(1)
	{
		vector<vector<cv::Point> >  all_contours;
		vector<vector<cv::Point> >      contours;
		//Mat elem = 
		//	getStructuringElement(MORPH_ELLIPSE , cvSize(3,3) , cv::Point(-1,-1) );
		//erode(img_bin , img_bin , elem , cv::Point(-1,-1) );
		//dilate(img_bin , img_bin , elem, cv::Point(-1,-1) );

		vector<Vec4i> hierarchy;
		//cv::findContours( img_bin, all_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		cv::findContours( img_bin, all_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		vector<vector<Point>>convex_contour(all_contours.size() );
		vector<Point> approx_poly;
		vector<int>  apdNum(all_contours.size() );
		vector<RotatedRect >rects;
		vector<double>sigmas(all_contours.size() );
		double mv = 99999; int uind = 0;

		//1.����͹��Ե
		for(int i=0;i<all_contours.size();i++){
			convexHull(all_contours[i] , convex_contour[i] , false , true);
			double ts = this->mLineDetector.isCircle(convex_contour[i]);
			sigmas[i] = ts;

			RotatedRect rr = minAreaRect(convex_contour[i]);
			rects.push_back(rr);
			approxPolyDP(convex_contour[i] , approx_poly , 1 , true);
			apdNum[i] = approx_poly.size();
			if(mv > ts && apdNum[i] >= 10){
				mv = ts;
				uind = i;
			}
		}

		char buf[255];

		//2.�ж����л�
		vector<vector<Point2f> >  arc_pts(0);
		vector<int>        ulist;
		vector<int>        uInds;
		//bool isFind = this->mLineDetector.findArc2(convex_contour[uInd] , arc_pts , ulist);
		for (int i=0;i< convex_contour.size();++i)
		{
			vector<Point2f>   arc_pt(0);
			bool isFind = this->mLineDetector.findArc2(convex_contour[i] , arc_pt , ulist);
			
			if (isFind)
			{
				arc_pts.push_back(arc_pt);
				uInds.push_back(i);
			}
		}

		/////////////////////////////////////////////////////
		//2.1 �жϻ�����϶�
		//Mat rvec, tvec;
		for (int i=0;i< arc_pts.size();++i)
		{
			sortPoints(arc_pts[i]);
			this->mLineDetector.sampleArc(arc_pts[i]);
		}
		
		//vector<Point>c2c_points;
		//////////////////////////////////////////////////////////////////////////
		//2.2 show the result
		for (int i=0;i<uInds.size();++i)
		{
			circle(frame , convex_contour[uInds[i] ][ulist[0] ], 5 , Scalar(0,0,255) , 1);
			circle(frame , convex_contour[uInds[i] ][ulist[1] ], 5 , Scalar(0,0,255) , 1);
		}

		//circle(frame , Point(lastCenter.x , lastCenter.y) , 5 , Scalar(0,0,255) , 1);
		for(int i=0;i<arc_pts.size();i++){
			for(int j=0;j<arc_pts.size();j++){
			circle(frame , Point(arc_pts[i][j].x , arc_pts[i][j].y) , 2 , Scalar(255,0,0) , 1);
			}
		}

		for (int i=0;i<uInds.size();++i)
		{
			sprintf(buf, "%d" , ulist[0] );
			putText(frame , buf, convex_contour[uInds[i] ][ulist[0]] , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
			sprintf(buf, "%d" , ulist[1] );
			putText(frame , buf, convex_contour[uInds[i] ][ulist[1]] , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
		}

		// 		for(int i=0;i<convex_contour.size();i++){
		// 			if(apdNum[i] <= 10) continue;
		for (int i=0;i<uInds.size();++i)
		{
			drawContours(frame , convex_contour , uInds[i] , Scalar(0,0,255));
		}
		//imshow("BinaryMast" , img_bin);

		//RotatedRect elp = fitEllipse(arc_pts);
		//ellipse(frame , elp , Scalar(255,0,0) , 2);

		//Point2f *arc_pts_b = new Point2f(4);
		//elp.points(arc_pts_b);

		//if(arc_pts_b[0].x <frame.cols && arc_pts_b[0].y < frame.rows
		//	&& arc_pts_b[1].x <frame.cols && arc_pts_b[1].y < frame.rows){
		//		line(frame  , 
		//			Point(arc_pts_b[0].x , arc_pts_b[0].y), 
		//			Point(arc_pts_b[1].x , arc_pts_b[1].y) , Scalar(255,0,0));
		//}
		//if(arc_pts_b[1].x <frame.cols && arc_pts_b[1].y < frame.rows
		//	&& arc_pts_b[2].x <frame.cols && arc_pts_b[2].y < frame.rows){
		//		line(frame  , 
		//			Point(arc_pts_b[1].x , arc_pts_b[1].y), 
		//			Point(arc_pts_b[2].x , arc_pts_b[2].y) , Scalar(255,0,0));
		//}

		//if(arc_pts_b[2].x <frame.cols && arc_pts_b[2].y < frame.rows
		//	&& arc_pts_b[3].x <frame.cols && arc_pts_b[3].y < frame.rows){
		//		line(frame  , 
		//			Point(arc_pts_b[2].x , arc_pts_b[2].y), 
		//			Point(arc_pts_b[3].x , arc_pts_b[3].y) , Scalar(255,0,0));
		//}

		//if(arc_pts_b[0].x <frame.cols && arc_pts_b[0].y < frame.rows
		//	&& arc_pts_b[3].x <frame.cols && arc_pts_b[3].y < frame.rows){
		//		line(frame  , 
		//			Point(arc_pts_b[0].x , arc_pts_b[0].y), 
		//			Point(arc_pts_b[3].x , arc_pts_b[3].y) , Scalar(255,0,0));
		//}

		imshow("Contour" , frame);
		waitKey(0);
	}

	return 0;
}


int CGlintRemover::DetectArc (int argc, _TCHAR* argv[])
{

	////std::ifstream fin("calibdata.txt");
	////std::ofstream fout("calibration_result.txt"); 
	////std::string filename;
	////std::vector<Mat >inputFrames;
	////while (std::getline(fin,filename))
	////{
	////	Mat t = imread(filename.c_str(),1);
	////	inputFrames.push_back(t);
	////}

	////camCalib myCamCalib;
	////MyCamera cam;
	////vector<Mat>rMat,tMat;
	////CvSize imgSize;
	////imgSize.width = inputFrames[0].cols;
	////imgSize.height = inputFrames[0].rows;
	////bool isDetected = myCamCalib.cameraCab(inputFrames , cam);
	//////bool isDetected = myCamCalib.cameraAutoCab(imgSize , cam);

	////if(!isDetected){
	////	std::cout<<"Failed to calib the camera"<<endl;
	////	return -1;
	////}

	////rec3D myRec3D(cam);

	////cv::VideoCapture cap("camera.avi");
	////if(!cap.isOpened()){
	////	std::cout<<"Failed to open video"<<endl;
	////	system("pause");
	////	return -1;
	////}

	//cv::Mat frame, frame1;
	//int fCount = 0;
	//cv::Point2f lastCenter;

	////while(cap.isOpened()){
	////if(fCount++ >= cap.get(CV_CAP_PROP_FRAME_COUNT)-1){
	////	break;
	////}

	//cap >> frame;
	//cv::Mat fgray, img_bin;
	//int thresh_size = 51;
	//cvtColor(frame , fgray , CV_RGB2GRAY);

	//cv::adaptiveThreshold(
	//	fgray , img_bin , 255 , cv::ADAPTIVE_THRESH_GAUSSIAN_C ,
	//	cv::THRESH_BINARY_INV  , thresh_size , thresh_size / 3);
	////threshold(fgray , img_bin , THRESH_BINARY_INV + THRESH_OTSU , 255 , 100);
	////imshow("Binary" , img_bin);
	////continue;
	//if(1){
	//	vector<vector<cv::Point> >all_contours;
	//	vector<vector<cv::Point> >contours;
	//	Mat elem = getStructuringElement(MORPH_ELLIPSE , cvSize(3,3) , cv::Point(-1,-1) );
	//	erode(img_bin , img_bin , elem , cv::Point(-1,-1) );
	//	dilate(img_bin , img_bin , elem, cv::Point(-1,-1) );
	//	vector<Vec4i> hierarchy;
	//	findContours( img_bin, all_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	//	vector<vector<Point>>convex_contour(all_contours.size());
	//	vector<Point> approx_poly;
	//	vector<int>  apdNum(all_contours.size());
	//	vector<RotatedRect >rects;
	//	vector<double>sigmas(all_contours.size());
	//	double mv = 99999; int uind = 0;
	//	for(int i=0;i<all_contours.size();i++){
	//		convexHull(all_contours[i] , convex_contour[i] , false , true);
	//		double ts = this->mLineDetector.isCircle(convex_contour[i]);
	//		sigmas[i] = ts;

	//		RotatedRect rr = minAreaRect(convex_contour[i]);
	//		rects.push_back(rr);
	//		approxPolyDP(convex_contour[i] , approx_poly , 1 , true);
	//		apdNum[i] = approx_poly.size();
	//		if(mv > ts && apdNum[i] >= 10){
	//			mv = ts;
	//			uind = i;
	//		}
	//	}
	//	char buf[255];
	//	int uInd = 0;
	//	if(fCount == 1){ // first frame
	//		uInd = uind;
	//	}
	//	else{
	//		Point2f * ppts = new Point2f(4);
	//		double minDis = 99999; uInd = 0;
	//		for(int i=0;i<all_contours.size();i++){
	//			if(apdNum[i] <= 10)
	//				continue;
	//			rects[i].points(ppts);
	//			Point2f mP(0,0);
	//			for(int k=0;k<4;k++){
	//				mP.x += ppts[k].x;
	//				mP.y += ppts[k].y;
	//			}
	//			mP.x /= 4; mP.y /= 4;
	//			double tD = sqrt((double)(mP.x - lastCenter.x)*(mP.x - lastCenter.x) + 
	//				(mP.y - lastCenter.y)*(mP.y - lastCenter.y));
	//			if(minDis > tD){
	//				minDis = tD;
	//				uInd = i;
	//			}
	//		}

	//	}


	//	//drawContours(frame , convex_contour , uInd , Scalar(0,0,255));
	//	Point2f *pts = new Point2f(4);
	//	rects[uInd].points(pts);
	//	// 		line(frame , pts[0] , pts[1] , Scalar(255,0,0) , 2);
	//	// 		line(frame , pts[1] , pts[2] , Scalar(255,0,0) , 2);
	//	// 		line(frame , pts[2] , pts[3] , Scalar(255,0,0) , 2);
	//	// 		line(frame , pts[3] , pts[0] , Scalar(255,0,0) , 2);

	//	Point2f mmP(0,0);
	//	Point2f * ppp = new Point2f(4);
	//	rects[uInd].points(ppp);
	//	for(int i=0;i<4;i++){
	//		mmP.x += ppp[i].x;
	//		mmP.y += ppp[i].y;
	//	}
	//	mmP.x /= 4;
	//	mmP.y /= 4;

	//	lastCenter.x = mmP.x;
	//	lastCenter.y = mmP.y;

	//	vector<Point2f>arc_pts;
	//	vector<int>ulist;
	//	bool isFind = 
	//		this->mLineDetector.findArc2(convex_contour[uInd] , arc_pts , ulist);

	//	if(!isFind ){
	//		continue;
	//	}

	//	/////////////////////////////////////////////////////
	//	// ok, calculate the 3D position
	//	Mat rvec, tvec;
	//	sortPoints(arc_pts);
	//	vector<Point>c2c_points;
	//	//projectConvex2Contour(arc_pts , all_contours[uInd] , c2c_points);

	//	// 			for(int i=0;i<arc_pts.size();i++){
	//	// 				circle(frame , Point(arc_pts[i].x,arc_pts[i].y) , 4, Scalar(0,0,255));
	//	// 			}
	//	this->mLineDetector.sampleArc(arc_pts);
	//	// 			for(int i=0;i<arc_pts.size();i++){
	//	// 				circle(frame , Point(arc_pts[i].x,arc_pts[i].y) , 2 , Scalar(255,0,0));
	//	// 			}
	//	// 			imshow("Debug" , frame);
	//	// 			if(waitKey() == 'q')
	//	// 				break;

	//	myRec3D.reconstruct3D(frame , arc_pts , rvec , tvec);

	//	//myRec3D.reconstruct3DWithCircleDebug(frame , arc_pts , rvec , tvec);

	//	//////////////////////////////////////////////////////////////////////////
	//	// show the result

	//	circle(frame , convex_contour[uInd][ulist[0]], 5 , Scalar(0,0,255) , 1);
	//	circle(frame , convex_contour[uInd][ulist[1]], 5 , Scalar(0,0,255) , 1);

	//	//circle(frame , Point(lastCenter.x , lastCenter.y) , 5 , Scalar(0,0,255) , 1);
	//	for(int i=0;i<arc_pts.size();i++){
	//		circle(frame , Point(arc_pts[i].x , arc_pts[i].y) , 2 , Scalar(255,0,0) , 1);
	//		//sprintf(buf, "%d" , arc_pts[i].ind);
	//		//putText(frame , buf, Point(arc_pts[i].pos.x , arc_pts[i].pos.y) , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);

	//	}

	//	sprintf(buf, "%d" , ulist[0]);
	//	putText(frame , buf, convex_contour[uInd][ulist[0]] , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
	//	sprintf(buf, "%d" , ulist[1]);
	//	putText(frame , buf, convex_contour[uInd][ulist[1]] , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);

	//	// 		for(int i=0;i<convex_contour.size();i++){
	//	// 			if(apdNum[i] <= 10) continue;
	//	drawContours(frame , convex_contour , uInd , Scalar(0,0,255));

	//	// 			sprintf(buf, "%d" , apdNum[i]);
	//	// 			//putText(frame , buf, convex_contour[i][0] , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
	//	// 			Point2f *pts = new Point2f(4);
	//	// 			rects[i].points(pts);
	//	// 			line(frame , pts[0] , pts[1] , Scalar(255,0,0) , 2);
	//	// 			line(frame , pts[1] , pts[2] , Scalar(255,0,0) , 2);
	//	// 			line(frame , pts[2] , pts[3] , Scalar(255,0,0) , 2);
	//	// 			line(frame , pts[3] , pts[0] , Scalar(255,0,0) , 2);
	//	// 			putText(frame , "2", Point(pts[1].x,pts[1].y) , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
	//	// 			putText(frame , "3", Point(pts[2].x,pts[2].y) , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
	//	// 
	//	// 			sprintf(buf, "%lf" , sigmas[i]);
	//	// 			putText(frame , buf, convex_contour[i][convex_contour[i].size() - 1] , FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
	//	// 
	//	// 			//rectangle(frame , rects[i].boundingRect() , Scalar(255,0,0) , 1);
	//	// 		}
	//	//imshow("BinaryMast" , img_bin);

	//	if(arc_pts.size() < 5) continue;
	//	RotatedRect elp = fitEllipse(arc_pts);
	//	ellipse(frame , elp , Scalar(255,0,0) , 2);
	//	Point2f *arc_pts_b = new Point2f(4);
	//	elp.points(arc_pts_b);
	//	if(arc_pts_b[0].x <frame.cols && arc_pts_b[0].y < frame.rows
	//		&& arc_pts_b[1].x <frame.cols && arc_pts_b[1].y < frame.rows){
	//			line(frame  , Point(arc_pts_b[0].x , arc_pts_b[0].y), 
	//				Point(arc_pts_b[1].x , arc_pts_b[1].y) , Scalar(255,0,0));
	//	}
	//	if(arc_pts_b[1].x <frame.cols && arc_pts_b[1].y < frame.rows
	//		&& arc_pts_b[2].x <frame.cols && arc_pts_b[2].y < frame.rows){
	//			line(frame  , Point(arc_pts_b[1].x , arc_pts_b[1].y), 
	//				Point(arc_pts_b[2].x , arc_pts_b[2].y) , Scalar(255,0,0));
	//	}

	//	if(arc_pts_b[2].x <frame.cols && arc_pts_b[2].y < frame.rows
	//		&& arc_pts_b[3].x <frame.cols && arc_pts_b[3].y < frame.rows){
	//			line(frame  , Point(arc_pts_b[2].x , arc_pts_b[2].y), 
	//				Point(arc_pts_b[3].x , arc_pts_b[3].y) , Scalar(255,0,0));
	//	}

	//	if(arc_pts_b[0].x <frame.cols && arc_pts_b[0].y < frame.rows
	//		&& arc_pts_b[3].x <frame.cols && arc_pts_b[3].y < frame.rows){
	//			line(frame  , Point(arc_pts_b[0].x , arc_pts_b[0].y), 
	//				Point(arc_pts_b[3].x , arc_pts_b[3].y) , Scalar(255,0,0));
	//	}
	//	//imshow("Contour" , frame);
	//	//	}

	//	//	//system("pause");
	//	//}

	//	
	//}
	return 0;
}