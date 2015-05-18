#include "HandGesture.h"
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>

HandGesture::HandGesture(void)
{
	interface = new pcl::OpenNIGrabber();

	boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant)> 
		f = boost::bind (&HandGesture::cloud_cb_imgs, this, _1, _2, _3);

	interface->registerCallback (f);
}


HandGesture::~HandGesture(void)
{
}

void HandGesture::run()
{	
	interface->start ();	

	while (!viewer.viewer->wasStopped())
	{		
		boost::this_thread::sleep (boost::posix_time::seconds (1));
	}

	interface->stop ();
}

void HandGesture::cloud_cb_imgs (const boost::shared_ptr<openni_wrapper::Image> &rgb, const boost::shared_ptr<openni_wrapper::DepthImage> &depth, float constant)
{	
//	FPS_CALC("cb_imgs");

	observer.setInput(rgb, depth);

	Param param;
	param.cameraAngle = 30;
    param.pointstep = 2;
    param.face_image_scale = 5;
	param.face_concent = 0.5;
	param.head_radius = 0.1;
	param.arm_length = 1;		 
	param.hand_length = 0.1;
    param.nTrajectory = 100;
	param.minDist = 0.02;	
    param.minHandPoints = 300;

    state = observer.run(param);

    // output
    Eigen::Vector3d handPos;
    vector<Eigen::Vector3d> handPosTraj;
    int handstate;
    Eigen::Vector3d facePos;
    Eigen::Vector3d headPos;
    int gesture;
    float scroll;

    // output display example
//    if(getHandPos(handPos))
//        cout<<"Hand at "<<handPos[0]<<" "<<handPos[1]<<" "<<handPos[2]<<endl;

//    if(getHandPosTraj(handPosTraj))
//        cout<<"Previous Hand at "<<handPosTraj.at(handPosTraj.size()-2)[0]<<" "<<handPosTraj.at(handPosTraj.size()-2)[1]<<" "<<handPosTraj.at(handPosTraj.size()-2)[2]<<endl;


    if(getHandState(handstate)){
        cout<<"Hand State: ";
        switch(handstate){
        case 0: cout<<"PALM"<<endl; break;
        case 1: cout<<"FIST"<<endl; break;
        }
    }


//    if(getFacePos(facePos))
//        cout<<"Face at "<<facePos[0]<<" "<<facePos[1]<<" "<<facePos[2]<<endl;

//    if(getHeadPos(headPos))
//        cout<<"Head at "<<headPos[0]<<" "<<headPos[1]<<" "<<headPos[2]<<endl;
/*
    if(getGesture(gesture)){
        cout<<"Gesture: ";
        switch(gesture){
        case 1: cout<<"LEFT"<<endl;   break;
        case 2: cout<<"RIGHT"<<endl;   break;
        case 3: cout<<"UP"<<endl;   break;
        case 4: cout<<"DOWN"<<endl;   break;
        case 5: cout<<"PUSH"<<endl;   break;
        case 6: cout<<"PULL"<<endl;   break;
        }
    }
*/
    if(getScroll(scroll)){
        cout<<"Scroll: "<<scroll<<endl;
    }

    showView(1); // 1:hand, 2:face, 3:head
}


bool HandGesture::getHandPos(Eigen::Vector3d &handPos)
{
    if(state.isHandDetected){
        handPos = state.handPos3d_filtered;
        return 1;
    }
    return 0;

}

bool HandGesture::getHandState(int &handstate)    // 0: PALM, 1: FIST
{
    if(state.isHandDetected){
        if(state.handState == PALM) handstate = 0;
        if(state.handState == FIST) handstate = 1;
        return 1;
    }
    return 0;
}

bool HandGesture::getHandPosTraj(vector<Eigen::Vector3d> &handPosTraj)
{
    if(state.isHandDetected && state.handTrajectory.size() > 2){
        handPosTraj = state.handTrajectory;
        return 1;
    }
    return 0;
}

bool HandGesture::getFacePos(Eigen::Vector3d &facePos)
{
    if(state.isFaceDetected){
        facePos = state.facePos3d;
        return 1;
    }
    return 0;
}

bool HandGesture::getHeadPos(Eigen::Vector3d &headPos)
{
    if(state.isHeadDetected){
        headPos = state.headPos3d;
        return 1;
    }
    return 0;
}

bool HandGesture::getGesture(int &gesture)
{
    if(state.gesture != NONE){
        switch(state.gesture){
        case LEFT:	gesture = 1;	break;
        case RIGHT:	gesture = 2;	break;
        case UP:	gesture = 3;		break;
        case DOWN:	gesture = 4;	break;
        case PUSH:	gesture = 5;	break;
        case PULL:	gesture = 6;	break;
        }
        return 1;
    }
    return 0;
}

bool HandGesture::getScroll(float &scroll)
{
    if(state.scroll != 0){
        scroll = state.scroll;
        return 1;
    }
    return 0;
}

void HandGesture::showView(int n)
{
    if(n == 1){
        if(state.isHandDetected)    viewer.showCloud(state.hand3d_filtered);
    }
    if(n == 2){
        if(state.isFaceDetected)    viewer.showCloud(state.face3d);
    }
    if(n == 3){
        if(state.isHeadDetected)    viewer.showCloud(state.head3d);
    }
}
