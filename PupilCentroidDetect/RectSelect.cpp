#include "stdafx.h"
#include "RectSelect.h"


CRectSelect::CRectSelect(void)
{
	////this->mNumMoveRect  = 8;     //获取移动框的个数： 共8个生成

	////一次性生成可以没有重复，真TMD巧合！！！
	//this->mNumRoateRect           = (4+1);        //获取旋转框的个数
	//this->mNumRoateResizeRect     = 5*2;      //获取旋转缩放框的个数 缩放比率为 1，1.1
	//this->mNumRoateResizeMoveRect = 5*2*9;    //获取旋转移动框的个数 移动为3个单元 9个生成： 共90个生成

	////this->mNumRoateMoveRect = 8*9; //获取旋转移动框的个数
	////this->mNumTotalRect =1+ this->mNumMoveRect  + this->mNumRoateResizeRect  + this->mNumRoateResizeMoveRect;    //获取旋转移动框的个数

	//this->mNumTotalRect = this->mNumRoateResizeMoveRect;    //获取旋转移动框的个数

	//this->mWidth =0;
	//this->mHeight=0;

	//this->mDetaAngle =5/180*M_PI;           //旋转间隔

	//this->mDetaMoveStep = 0.03;           //移动单位个数
	//this->mDetaMovePixels = this->mDetaMoveStep*this->mWidth;           //移动像素间隔

}

CRectSelect::CRectSelect(cv::Mat &Image)
{

	//一次性生成可以没有重复，真TMD巧合！！！
	this->mNumRoateRect           = (4+1);        //获取旋转框的个数
	this->mNumRoateResizeRect     = 5*2;      //获取旋转缩放框的个数 缩放比率为 1，1.1
	this->mNumRoateResizeMoveRect = 5*2*9;    //获取旋转移动框的个数 移动为3个单元 9个生成： 共90个生成

	this->mNumTotalRect           = this->mNumRoateResizeMoveRect;    //获取旋转移动框的个数

	this->mWidth  =  Image.cols;
	this->mHeight =  Image.rows;

	this->mRadius    = 1;
	this->mRadiusS  =100;
	this->mDetaAngle = 5/180*M_PI;           //旋转间隔

	//this->mDetaMoveStep = 0.03;           //移动单位个数
	//this->mDetaMovePixels = this->mDetaMoveStep*this->mWidth;           //移动像素间隔
	this->mDetaMoveStep  = 0.05; //移动单位个数
	this->mDetaMovePixels=    0;  //默认为0个

	//手动获取瞳孔中心坐标
	this->mPupilCenter.x =508;
	this->mPupilCenter.y =513;

	//CvPoint  pupilCenter;
	//pupilCenter.x =508;
	//pupilCenter.y =513;

	this->mImage =Image.clone();


	//初始化框和初始化框图像
	this->mRectRoate.resize(0);
	this->mRectRoateResize.resize(0);
	this->mRectRoateResizeMove.resize(0);

	this->mZoomFactor = 1.1;
	this->mImageRect.resize(0);
	this->mImageRectRoate.resize(0);
	this->mImageRectRoateResize.resize(0);
	this->mImageRectRoateResizeMove.resize(0);
	this->mCenterRoateResizeMove.resize(0);
	
	return ;
}

CRectSelect::CRectSelect(int width,int height)
{
	//this->mNumMoveRect  = 8;     //获取移动框的个数： 共8个生成

	//一次性生成可以没有重复，真TMD巧合！！！
	this->mNumRoateRect           = (4+1);        //获取旋转框的个数
	this->mNumRoateResizeRect     = 5*2;      //获取旋转缩放框的个数 缩放比率为 1，1.1
	this->mNumRoateResizeMoveRect = 5*2*9;    //获取旋转移动框的个数 移动为3个单元 9个生成： 共90个生成

	this->mNumTotalRect           = this->mNumRoateResizeMoveRect;    //获取旋转移动框的个数

	this->mWidth  =  width;
	this->mHeight = height;

	this->mRadius    = 1;
	this->mDetaAngle = 5/180*M_PI;           //旋转间隔

	this->mDetaMoveStep = 0.03;           //移动单位个数
	this->mDetaMovePixels = this->mDetaMoveStep*this->mWidth;           //移动像素间隔

	//手动获取瞳孔中心坐标
	this->mPupilCenter.x =0;
	this->mPupilCenter.y =0;

	//初始化框和初始化框图像
	this->mRectRoate.resize(0);
	this->mRectRoateResize.resize(0);
	this->mRectRoateResizeMove.resize(0);

	this->mImageRectRoate.resize(0);
	this->mImageRectRoateResize.resize(0);
	this->mImageRectRoateResizeMove.resize(0);

}

CRectSelect::~CRectSelect(void)
{


}


int CRectSelect::GetPicPatches()
{


	return 1;
}
int CRectSelect::GetPicPatches(std::vector<cv::Mat> &patchList)
{
	//int radius     =  100;
	//this->mRadiusS =radius

	//获取5个旋转框
	//this->SelectRoateRect(radius,pupilCenter);
	this->SelectRoatePatch(this->mRadiusS,this->mPupilCenter);

	//获取10个缩放框
	this->SelectRoateResizePatch(this->mRadiusS,this->mPupilCenter);
	
	//获取90个平移框
	this->SelectRoateResizeMovePatch(this->mRadiusS,this->mPupilCenter);

	return 1;
}


//选取旋转框
int CRectSelect::SelectRoateRect()
{
	double detaAngle =this->mDetaAngle;



	return 1;
}
//选取旋转框
int CRectSelect::SelectRoateRect(int radius)
{
	
	this->mRadius =radius;
	double detaAngle =this->mDetaAngle;

	return 1;
}

//选取旋转框
int CRectSelect::SelectRoateRect(int radius,CvPoint  &pupilCenter)
{

	this->mRadius =radius;
	this->mPupilCenter.x =pupilCenter.x;
	this->mPupilCenter.y =pupilCenter.y;

	double detaAngle =this->mDetaAngle;


	this->mRectRoate.resize(0);

	//1.获取基本矩形框--方形框
	this->mRectHeight = radius*2 +1;
	this->mRectWidth  = radius*2 +1;
	//this->mRectHeight = pupilCenter.x + radius;
	//this->mRectWidth  = pupilCenter.y + radius;

	//this->mRectRoate.resize(5);
	CvRect rectBasic;
	cv::Mat rectImgBasic;
	rectBasic.x = pupilCenter.x -radius;
	rectBasic.y = pupilCenter.y -radius;
	rectBasic.height = this->mRectHeight;
	rectBasic.width  =  this->mRectWidth;

	
	//2.获取旋转矩形框--方形框
	//获取旋转框必须深入到图像，否则必须进行两次转换
	//用几个循环取代

	//CvRect rotateBasicR1,rotateBasicR2,rotateBasicL1,rotateBasicL2;
	//rotateBasicR1.x = pupilCenter.x -radius;
	//rotateBasicR1.y = pupilCenter.y -radius;
	//rotateBasicR2.x = pupilCenter.x -radius;
	//rotateBasicR2.y = pupilCenter.y -radius;
	//rotateBasicL1.x = pupilCenter.x -radius;
	//rotateBasicL1.y = pupilCenter.y -radius;
	//rotateBasicL2.x = pupilCenter.x -radius;
	//rotateBasicL2.y = pupilCenter.y -radius;

	double angle =  5;  // 旋转角度5度  //double angle = 30;  // 旋转角度  
	double scale =  1; // 缩放尺度      //double scale = 0.5; // 缩放尺度

	//cv::Mat rotateMatR1,rotateMatR2,rotateMatL1,rotateMatL2;   
	//rotateMatR1 = cv::getRotationMatrix2D(center, angle, scale); 

	//cv::Mat rotateImgR1,rotateImgR2,rotateImgL1,rotateImgL2;  
	//cv::warpAffine(rectImgBasic, rotateImgR1, rotateMatR1, rectImgBasic.size() ); 

	//this->mRectRoate.push_back(rectBasic);

	std::vector<double> angles(5);
	//angles[0] =-10/180*M_PI;
	//angles[1] = -5/180*M_PI;
	//angles[2] =  0/180*M_PI;
	//angles[3] =  5/180*M_PI;
	//angles[4] = 10/180*M_PI;

	//angles[0] =-60/180*M_PI;
	//angles[1] = -30/180*M_PI;
	//angles[2] =  0/180*M_PI;
	//angles[3] =  30/180*M_PI;
	//angles[4] = 60/180*M_PI;

	//angles[0] =-60;
	//angles[1] = -30;
	//angles[2] =  0;
	//angles[3] =  30;
	//angles[4] = 60;

	angles[0] =-10;
	angles[1] = -5;
	angles[2] =  0;
	angles[3] =  5;
	angles[4] = 10;
	
	std::vector<cv::Mat> rotateMats(5);
	std::vector<cv::Mat> imageRect(5);
	std::vector<cv::Mat> rotateSrcImgs(5);
	std::vector<cv::Mat> rotateImgs(5);

    //获取旋转图像
	CvRect rectT;
	rectT.height =this->mRectHeight *2;
	rectT.width  = this->mRectWidth *2;
	rectT.x      = pupilCenter.x - radius*2;
	rectT.y      = pupilCenter.y - radius*2;

	CvPoint pStart;
	pStart.x = rectT.x ;
	pStart.y = rectT.y ; 
	CvMat   * pRect =cvCreateMat(rectT.height,rectT.width,CV_8UC1);

	//cv::Point2f center = cv::Point2f(pupilCenter.x, pupilCenter.y);  // 旋转中心  
	cv::Point2f center = cv::Point2f(rectT.width/2, rectT.height/2);  // 注意 旋转中心 

	for (int i=0; i<angles.size(); ++i)
	{
		//图像中心点为旋转中心
		//rotateMats[i] =cv::getRotationMatrix2D(center, angles[i], scale); 
		//cv::warpAffine(rectImgBasic, rotateImgs[i], rotateMats[i], rectImgBasic.size() );
		//cv::warpAffine(this->mImage, rotateSrcImgs[i], rotateMats[i], rectImgBasic.size() );

		rotateMats[i] =cv::getRotationMatrix2D(center, angles[i], scale); 
		IplImage  imgT  = (this->mImage);
		IplImage* pImgT = &imgT;
		cvGetSubRect(pImgT,pRect,rectT);
		cv::Mat rectM(pRect);

		////cv::Mat rectNormal(rectT.height,rectT.width,CV_8UC3);
		////this->RerankXY(rectM,rectNormal,pStart);

		//cv::imshow("rotateSrcImgs[i]",rectM);cv::waitKey(0);

		//仿射变换前必须对下标重排列
		cv::warpAffine(rectM, rotateSrcImgs[i], rotateMats[i], rectImgBasic.size() );
		//cv::imshow("rotateSrcImgs[i]",rotateSrcImgs[i]);cv::waitKey(0);
		//cvReleaseImage(&pImgT);//create 对应release
	}

	//再次截取中心位置
	CvRect rectF;
	rectF.height = this->mRectHeight  ;
	rectF.width  = this->mRectWidth   ;
	rectF.x      = this->mRectWidth /2;
	rectF.y      = this->mRectHeight/2;

	for (int i=0; i<rotateSrcImgs.size(); ++i)
	{
		IplImage  imgT  = ( rotateSrcImgs[i] );
		IplImage* pImgT = &imgT;
		cvGetSubRect( pImgT, pRect, rectF);
		cv::Mat rectM( pRect);

		//cv::imshow("rotateSrcImgs[i]",rectM);cv::waitKey(0);
		this->mImageRectRoate.push_back(rectM);
		//cvReleaseImage(&pImgT);
	}

	return 1;
}

//选取旋转框框定的图像
//生成5个旋转矩阵，存储在 this->mImageRectRoate 里面
//同时生成5个二倍大小的矩阵，存储在mImageRectRoate2 里面
int CRectSelect::SelectRoatePatch(int radius,CvPoint  &pupilCenter)
{
	this->mRadius =radius;
	this->mPupilCenter.x =pupilCenter.x;
	this->mPupilCenter.y =pupilCenter.y;

	double detaAngle =this->mDetaAngle;

	this->mRectRoate.resize(0);

	//1.获取基本矩形框--方形框
	this->mRectHeight = radius*2 +1;
	this->mRectWidth  = radius*2 +1;

	CvRect rectBasic;
	cv::Mat rectImgBasic;
	this->mNormalRect.x =
		rectBasic.x = pupilCenter.x -radius;
	this->mNormalRect.y =
		rectBasic.y = pupilCenter.y -radius;
	this->mNormalRect.height =
		rectBasic.height = this->mRectHeight;
	this->mNormalRect.width =
		rectBasic.width  =  this->mRectWidth;

	//2.获取旋转矩形框--方形框
	//获取旋转框必须深入到图像，否则必须进行两次转换


	double angle =  5;  // 旋转角度5度  //double angle = 30;  // 旋转角度  
	double scale =  1; // 缩放尺度      //double scale = 0.5; // 缩放尺度

	std::vector<double> angles(5);
	angles[0] =-10;
	angles[1] = -5;
	angles[2] =  0;
	angles[3] =  5;
	angles[4] = 10;

	std::vector<cv::Mat> rotateMats(5);
	std::vector<cv::Mat> imageRect(5);
	//std::vector<cv::Mat> rotateSrcImgs(5);
	this->mImageRectRoate2.resize(5);
	std::vector<cv::Mat> rotateImgs(5);

	//获取旋转图像
	CvRect rectT;
	mDoubleRect.height = 
		rectT.height =this->mRectHeight *2;
	mDoubleRect.width = 
		rectT.width  = this->mRectWidth *2;
	mDoubleRect.x = 
		rectT.x      = pupilCenter.x - radius*2;
	mDoubleRect.y = 
		rectT.y      = pupilCenter.y - radius*2;

	CvPoint pStart;
	pStart.x = rectT.x ;
	pStart.y = rectT.y ; 
	CvMat   * pRect =cvCreateMat(rectT.height,rectT.width,CV_8UC1);

	//cv::Point2f center = cv::Point2f(pupilCenter.x, pupilCenter.y);  // 旋转中心  
	cv::Point2f center = cv::Point2f(rectT.width/2, rectT.height/2);  // 注意 旋转中心 

	for (int i=0; i<angles.size(); ++i)
	{
		//图像中心点为旋转中心
		//rotateMats[i] =cv::getRotationMatrix2D(center, angles[i], scale); 
		//cv::warpAffine(rectImgBasic, rotateImgs[i], rotateMats[i], rectImgBasic.size() );
		//cv::warpAffine(this->mImage, rotateSrcImgs[i], rotateMats[i], rectImgBasic.size() );

		rotateMats[i] =cv::getRotationMatrix2D(center, angles[i], scale); 
		IplImage  imgT  = (this->mImage);
		IplImage* pImgT = &imgT;
		cvGetSubRect(pImgT,pRect,rectT);
		cv::Mat rectM(pRect);
		//cv::imshow("rotateSrcImgs[i]",rectM);cv::waitKey(0);

		//直接填入二倍的框图向量集合this->mImageRectRoate2
		cv::warpAffine(rectM, this->mImageRectRoate2[i], rotateMats[i], rectImgBasic.size() );
		//cv::imshow("rotateSrcImgs[i]",rotateSrcImgs[i]);cv::waitKey(0);
	}

	//再次截取中心位置
	CvRect rectF;
	rectF.height = this->mRectHeight  ;
	rectF.width  = this->mRectWidth   ;
	rectF.x      = this->mRectWidth /2;
	rectF.y      = this->mRectHeight/2;

	for (int i=0; i<this->mImageRectRoate2.size(); ++i)
	{
		IplImage  imgT  = ( this->mImageRectRoate2[i] );
		IplImage* pImgT = &imgT;
		cvGetSubRect( pImgT, pRect, rectF);
		cv::Mat rectM( pRect);

		//cv::imshow("rotateSrcImgs[i]",rectM);cv::waitKey(0);
		this->mImageRectRoate.push_back(rectM);
	}

	return this->mImageRectRoate.size();
}

//选取旋转缩放框
int CRectSelect::SelectRoateResizeRect()
{


	return 1;
}

//选取旋转缩放框
//生成10个图片，存储在mImageRectRoateResize 里面
//保存在mImageRectRoateResize2;//生成10个大2倍的图像块
int CRectSelect::SelectRoateResizePatch(int radius,CvPoint  &pupilCenter)
{

	//获取缩放图像
	//this->mImageRectRoateResize2;
	CvRect zoomRectT;
	zoomRectT.x = ( this->mDoubleRect.height * (this->mZoomFactor-1) )/2 ;
	zoomRectT.y = ( this->mDoubleRect.width  * (this->mZoomFactor-1) )/2 ;
	zoomRectT.height = this->mDoubleRect.height;
	zoomRectT.width  = this->mDoubleRect.width ;

	for (int k=0;k< this->mImageRectRoate2.size();++k)
	{
		cv::Mat RectMatSrc = this->mImageRectRoate2[k].clone();

		//1.获取缩放，然后再截取中心
		cv::Mat zoomMat;
		cv::resize(RectMatSrc,zoomMat,
			cv::Size(RectMatSrc.rows*this->mZoomFactor,RectMatSrc.cols*this->mZoomFactor),0,0, cv::INTER_AREA);

		//2.获取缩放框的中心
		CvMat   * pRect =cvCreateMat( this->mDoubleRect.height,this->mDoubleRect.width,CV_8UC1);
		IplImage  imgT  = zoomMat;
		IplImage* pImgT =   &imgT;
		cvGetSubRect( pImgT, pRect,zoomRectT);
		cv::Mat rectM( pRect);
		cv::Mat rectM2 =rectM.clone();

		this->mImageRectRoateResize2.push_back(RectMatSrc);
		this->mImageRectRoateResize2.push_back(rectM2);
		//cv::imshow("RectMatSrc",RectMatSrc);
		//cv::imshow("RectM",rectM);
		//cv::waitKey(0);
	}
	for (int i=0;i< this->mImageRectRoateResize2.size();++i )
	{
		cv::imshow("mImageRectRoateResize2[i]",this->mImageRectRoateResize2[i]);
		cv::waitKey(0);
	}

	return this->mImageRectRoateResize2.size();
}

//选取旋转缩放移动框
int CRectSelect::SelectRoateResizeMoveRect()
{


	return 1;
}

//选取旋转缩放移动框图片
//生成90个图片，存储在mImageRectRoateResizeMove 里面
int CRectSelect::SelectRoateResizeMovePatch(int radius,CvPoint  &pupilCenter)
{
	//移动单位个数
	this->mDetaMovePixelsX =this->mDetaMoveStep * this->mImageRectRoate[0].rows;    
	this->mDetaMovePixelsY =this->mDetaMoveStep * this->mImageRectRoate[0].rows;

	//获取平移图像
	//CvRect rectT;
	//rectT.height = this->mRectHeight *2;
	//rectT.width  = this->mRectWidth  *2;
	//rectT.x      = pupilCenter.x - radius*2;
	//rectT.y      = pupilCenter.y - radius*2;

	//再次截取中心位置


	//获取9个框
	this->mRectRoateResizeMove.resize(0);
	this->mCenterRoateResizeMove.resize(90);
	int idx =0;
	for (int i=-1;i<2;++i)
	{
		for (int j=-1; j<2; ++j)
		{
			idx =(i+1)*3+j+1;
			CvRect rectF;
			rectF.height = this->mRectHeight  ;
			rectF.width  = this->mRectWidth   ;

			rectF.x      = this->mRectWidth /2 + i*this->mDetaMovePixelsX;
			rectF.y      = this->mRectHeight/2 + j*this->mDetaMovePixelsY;
			this->mRectRoateResizeMove.push_back( rectF);
			this->mCenterRoateResizeMove[idx].x =  rectF.width /2 - i*this->mDetaMovePixelsX;
			this->mCenterRoateResizeMove[idx].y =  rectF.height/2 - j*this->mDetaMovePixelsY;
		}
	}

	idx = 0;
	for (int j=0;j< 10;++j )
	{
		for (int i=0;i< 9;++i )
		{
			idx =j*9 +i;
			this->mCenterRoateResizeMove[idx].x = this->mCenterRoateResizeMove[i].x;
			this->mCenterRoateResizeMove[idx].y = this->mCenterRoateResizeMove[i].y;
		}

	}

	//逐个缩放
	std::vector<cv::Mat>  moveMats;
	CvMat   * pRect = cvCreateMat(
		this->mNormalRect.height,this->mNormalRect.width,CV_8UC1);

	for (int k=0;k< this->mImageRectRoateResize2.size();++k)
	{
		//cv::Mat RectMat = this->mImageRectRoate2[k];
		IplImage  imgT  = this->mImageRectRoateResize2[k];
		//生成9个平移图片块
		for (int i=0;i<this->mRectRoateResizeMove.size();++i)
		{
				IplImage* pImgT = &imgT;
				cvGetSubRect( pImgT, pRect, this->mRectRoateResizeMove[i]);
				cv::Mat rectM( pRect);
				cv::Mat rectM2 =rectM.clone();
				//this->mImageRectRoateResizeMove.push_back(rectM);
				this->mImageRectRoateResizeMove.push_back(rectM2);
		}
	}


	
	cv::Point Center;
	Center.x = this->mImageRectRoateResizeMove[0].cols /2;
	Center.y = this->mImageRectRoateResizeMove[0].rows /2;
	int Length =8;
	cv::Scalar Color(0, 255, 0);
	int Width =2;
	int CV_A_Type = CV_AA;
	int Mark =0;

	const std::string FlagC = "E:/DataBase/face/examples/Flag"; 
	std::strstream   SFlagC; 
	std::string  SNum;

	for (int i=0;i< this->mImageRectRoateResizeMove.size();++i )
	{
		this->CvDrawCrossCursor(
			this->mImageRectRoateResizeMove[i],
			this->mCenterRoateResizeMove[i],
			Length,Color,Width,CV_A_Type, Mark);

		SFlagC.clear();
		SNum.clear();

		SFlagC<< i;
		SFlagC>> SNum;

		std::string Flag = FlagC;
		Flag.append(SNum);
		Flag.append(".png");

		cv::imwrite(Flag.c_str(),this->mImageRectRoateResizeMove[i]);
		//cv::imshow("this->mImageRectRoateResizeMove[i]",this->mImageRectRoateResizeMove[i]);
		//cv::waitKey(0);
	}

	return 1;
}


bool CRectSelect::RerankXY(cv::Mat &rectM, cv::Mat &rectNormal,CvPoint &pStart)
{
	for (int i=0;i<rectNormal.rows;++i)
	{
		for (int j=0;j<rectNormal.cols;++j)
		{
			rectNormal.at<cv::Vec3b>(j,i) =rectM.at<cv::Vec3b>(j,i);
			//rectNormal.at<uchar>(j,i) =rectM.at<uchar>(j+pStart.x,i+pStart.y);
			//rectNormal.at<cv::Vec3b>(j,i) =rectM.at<cv::Vec3b>(j+pStart.x,i+pStart.y);
		}
	}

	return 1;
}

//int CPlot::cvDrawCrossCursor(
int CRectSelect::CvDrawCrossCursor(
	cv::Mat &Canvas,cv::Point &Center,int Length,cv::Scalar &Color,int Width,int CV_A_Type,int Mark)
{
	int H = Length/2;
	cv::Point PointS;cv::Point PointE;
	PointS.x =Center.x ;
	PointS.y =Center.y -H;
	PointE.x =Center.x ;
	PointE.y =Center.y +H;

	cv::line(Canvas,PointS,PointE,Color,Width,CV_A_Type,Mark);

	PointS.x =Center.x -H;
	PointS.y =Center.y;
	PointE.x =Center.x +H;
	PointE.y =Center.y;
	cv::line(Canvas,PointS,PointE,Color,Width,CV_A_Type,Mark);

	return 1;
}