#pragma once

#include "common.h"
#include "FaceDetection.h"
#include "SkinDetection.h"
#include <pcl/io/openni_grabber.h>


struct Param {
	float cameraAngle;
	int pointstep;
    float face_image_scale;
	float face_concent;
	float head_radius;
	float arm_length;	// width, height, depth	
	float hand_length;
	int nTrajectory;	// 10fps
	float minDist;
    int minHandPoints;
};

typedef enum{PALM, FIST} HANDSTATE;
typedef enum{NONE, LEFT, RIGHT, UP, DOWN, PUSH, PULL} GESTURE;
struct State{
	bool isFaceDetected;
	cv::Rect face2d;
	CloudPtr face3d;
	Eigen::Vector3d facePos3d;

	bool isHeadDetected;
	Eigen::Vector3d headPos3d;
	CloudPtr head3d;

	bool isHandDetected;
	CloudPtr handLeft3d;
	CloudPtr handRight3d;
	Eigen::Vector3d handLeftPos3d;
	Eigen::Vector3d handRightPos3d;

	CloudPtr hand3d;
	Eigen::Vector3d handPos3d;
	Eigen::Vector3d handPos3d_farMost;
	CloudPtr hand3d_filtered;
	Eigen::Vector3d handPos3d_filtered;
	HANDSTATE handState;
	vector<Eigen::Vector3d> handTrajectory;

	GESTURE gesture;    
    float scroll;

	bool timeover;
};


class StateObserver
{

public:
	StateObserver(void);
	~StateObserver(void);

	void setInput(const boost::shared_ptr<openni_wrapper::Image> &rgb, const boost::shared_ptr<openni_wrapper::DepthImage> &depth);
	CloudPtr getCloudRaw(){return cloudRawPtr;};

	State run(Param param);

private:
	void convertToXYZRGBPointCloud (const boost::shared_ptr<cv::Mat> &image, const XnDepthPixel* depth_map, CloudPtr &cloud);

    void observeFace(float cameraAngle, float face_image_scale, float face_concent);
	float extractFace3D(float cameraAngle, const boost::shared_ptr<cv::Mat> &image, const XnDepthPixel* depth_map, cv::Rect face, CloudPtr &cloud, Eigen::Vector3d &pos, float concent);

	void observeHeadHand(float cameraAngle, float head_radius, float arm_length, int pointstep);
	void extractHeadHand3D(float cameraAngle, const boost::shared_ptr<cv::Mat> &image, const XnDepthPixel* depth_map, float head_radius, float arm_length, int pointstep);
    void handPCFiltering(float hand_length, int minHandPoints);
	void observeHandState(float hand_length);
	void observeHandTrajectory(int nTrajectory, float minDist);
	void createHandTrajectoryPC();

	void detectGesture(float minDist);
    void detectScroll();

private:
	FaceDetection faceDetection;

	boost::shared_ptr<cv::Mat> rgbImage;
	const XnDepthPixel* depth_map;

	State state;

	CloudPtr cloudRawPtr;	

	unsigned int rgb_frame_id_;
	unsigned int image_height_;
	unsigned int depth_height_;
	unsigned int image_width_;
	unsigned int depth_width_;
	unsigned int depth_max_;
	float focal_;	
	int centerX;
	int centerY;
	float constant;
	float XtoZ;
	float YtoZ;
	float bad_point;

};

