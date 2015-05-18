#pragma once

#include "Viewer.h"
#include "StateObserver.h"


class HandGesture
{
public:
	HandGesture(void);
	~HandGesture(void);

	void cloud_cb_imgs (const boost::shared_ptr<openni_wrapper::Image> &rgb, const boost::shared_ptr<openni_wrapper::DepthImage> &depth, float constant);
	void run();

    bool getHandPos(Eigen::Vector3d &handPos);
    bool getHandState(int &handstate);    // 0: PALM, 1: FIST
    bool getHandPosTraj(vector<Eigen::Vector3d> &handPosTraj);
    bool getFacePos(Eigen::Vector3d &facePos);
    bool getHeadPos(Eigen::Vector3d &facePos);
    bool getGesture(int &gesture);
    bool getScroll(float &scroll);

private:
    void showView(int n);

private:
	pcl::Grabber* interface;	
    Viewer viewer;
    State state;

	StateObserver observer;
};

