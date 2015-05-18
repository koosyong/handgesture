#include "SkinDetection.h"


SkinDetection::SkinDetection(void)
{
}


SkinDetection::~SkinDetection(void)
{
}

bool SkinDetection::isSkin(cv::Vec3b pixel)
{
	cv::Vec3b pix_bgr = pixel;

	int B = pixel.val[0];
	int G = pixel.val[1];
	int R = pixel.val[2];
	// apply rgb rule
	bool a = R1(R,G,B);

	int delta = 128;
	int Y = 0.299 * R + 0.587 * G + 0.114 * B;
	int Cr = (R-Y) * 0.713 + delta;
	int Cb = (B-Y) * 0.564 + delta;	
	bool b = R2(Y,Cr,Cb);
	//return 1;

	if(!b)
		return 1;
	else 
		return 0;
}