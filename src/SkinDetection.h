#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using std::cout;
using std::endl;

class SkinDetection
{
public:
	SkinDetection(void);
	~SkinDetection(void);

	static bool isSkin(cv::Vec3b pixel);
	
private:
	static bool R1(int R, int G, int B) {
		bool e1 = (R>95) && (G>40) && (B>20) && ((cv::max(R,cv::max(G,B)) - cv::min(R, cv::min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
		bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
		return (e1||e2);
	};

	static bool R2(float Y, float Cr, float Cb) {
		bool e3 = Cr <= 1.5862*Cb+20;
		bool e4 = Cr >= 0.3448*Cb+76.2069;
		bool e5 = Cr >= -4.5652*Cb+234.5652;
		bool e6 = Cr <= -1.15*Cb+301.75;
		bool e7 = Cr <= -2.2857*Cb+432.85;
		return e3 && e4 && e5 && e6 && e7;
	};

	static bool R2_2(float Y, float Cr, float Cb){
		bool e1 = (Y>=0) && (Y<=255);
		bool e2 = (Cr>=133) && (Cr<=173);
		bool e3 = (Cb>=77) && (Cb<=127);
		return e1 && e2 && e3;
	}

	static bool R3(float H, float S, float V) {
		return (H<25) || (H > 230);
	};
};

