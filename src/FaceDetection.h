#pragma once

#include "common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv_modules.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>

class FaceDetection
{
public:
	FaceDetection(void);
	~FaceDetection(void);

    cv::Rect detectFace(const cv::Mat rgbImage, float face_image_scale_);

private:
	cv::Rect findFace(const cv::Mat& grayImage);

	cv::CascadeClassifier frontal_face_cascade;
	cv::CascadeClassifier profile_face_cascade;
	cv::Mat grayImage;

    float face_image_scale;
};

