#include "FaceDetection.h"


FaceDetection::FaceDetection(void)
{
    string haarcascade_dir = "/home/koosy/Work/gesture/haarcascades/";
	string frontalFaceCascadeFilename = haarcascade_dir + "haarcascade_frontalface_alt.xml";
	string profileFaceCascadeFilename = haarcascade_dir + "haarcascade_profileface.xml";
	if(!frontal_face_cascade.load(frontalFaceCascadeFilename) || !profile_face_cascade.load(profileFaceCascadeFilename)) {
		cerr << "Error while loading HAAR cascades." << endl;
	}
}


FaceDetection::~FaceDetection(void)
{
}

cv::Rect FaceDetection::detectFace(const cv::Mat rgbImage, float face_image_scale_)
{
    face_image_scale = face_image_scale_;
	// Convert to grayscale
	cv::Mat grayImage;
    cv::cvtColor(rgbImage, grayImage, CV_BGR2GRAY);
	// Apply Histogram Equalization
	cv::equalizeHist(grayImage, grayImage);

	// Find a face and displace it
    cv::Rect face = findFace(grayImage);

    return face;


}

cv::Rect FaceDetection::findFace(const cv::Mat& grayImage) {
	// Detect frontal and side faces
	std::vector<cv::Rect> frontal_face_vector;
	std::vector<cv::Rect> profile_face_vector;
	std::vector<cv::Rect> face_vector;
    frontal_face_cascade.detectMultiScale(grayImage, frontal_face_vector, face_image_scale, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(25, 25));
    profile_face_cascade.detectMultiScale(grayImage, profile_face_vector, face_image_scale, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(25, 25));
	face_vector.insert(face_vector.end(), frontal_face_vector.begin(), frontal_face_vector.end());
	face_vector.insert(face_vector.end(), profile_face_vector.begin(), profile_face_vector.end());

	// Select the biggest face
	int max_area = 0;
	cv::Rect* face = new cv::Rect(0.0f, 0.0f, 0.0f, 0.0f);
	for(std::vector<cv::Rect>::iterator it = face_vector.begin(); it != face_vector.end(); it++) {
		if(std::max(max_area, it->width * it->height) != max_area) {
			face = &(*it);      
			max_area = it->width * it->height;
		}
	}

	return *face;
}
