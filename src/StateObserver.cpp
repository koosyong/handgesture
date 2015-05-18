#include "StateObserver.h"


StateObserver::StateObserver(void)
{	
	cloudRawPtr.reset(new Cloud);	

	// initialize state
	state.isFaceDetected = false;
	state.isHeadDetected = false;
	state.isHandDetected = false;
	state.timeover = false;
	state.head3d.reset(new Cloud);
}


StateObserver::~StateObserver(void)
{
}

void StateObserver::setInput(const boost::shared_ptr<openni_wrapper::Image> &rgb, const boost::shared_ptr<openni_wrapper::DepthImage> &depth)
{	
	unsigned int height = rgb->getHeight(); 
	unsigned int width = rgb->getWidth(); 	

	rgbImage.reset (new cv::Mat(height, width, CV_8UC3)); 	//CV_8UC3	
	rgb->fillRGB(rgbImage->cols, rgbImage->rows, rgbImage->data, rgbImage->step);

	depth_map = depth->getDepthMetaData().Data();

	unsigned int rgb_frame_id_ = rgb->getFrameID();
	unsigned int image_height_ = rgb->getHeight();
	unsigned int depth_height_ = depth->getHeight();
	unsigned int image_width_ = rgb->getWidth();
	unsigned int depth_width_ = depth->getWidth();
	unsigned int depth_max_ = depth->getDepthMetaData().ZRes();
	float focal_ = depth->getFocalLength();	

	// set output data	
	cloudRawPtr.reset(new Cloud);
	cloudRawPtr->header.frame_id = rgb_frame_id_;
	cloudRawPtr->height = std::max (image_height_, depth_height_);
	cloudRawPtr->width = std::max (image_width_, depth_width_);
	cloudRawPtr->is_dense = false;
	cloudRawPtr->points.resize (cloudRawPtr->height * cloudRawPtr->width);
	centerX = (cloudRawPtr->width >> 1);
	centerY = (cloudRawPtr->height >> 1);
	constant = 1.0f / focal_;
	XtoZ = atan(focal_/(2*cloudRawPtr->width))*2;
	YtoZ = atan(focal_/(2*cloudRawPtr->height))*2;
	bad_point = std::numeric_limits<float>::quiet_NaN();


	//XtoZ = constant;
	//YtoZ = constant;
}

State StateObserver::run(Param param)
{
    observeFace(param.cameraAngle, param.face_image_scale, param.face_concent);
	observeHeadHand(param.cameraAngle, param.head_radius, param.arm_length, param.pointstep);
	if(state.isHandDetected){
        handPCFiltering(param.hand_length, param.minHandPoints);
		observeHandState(param.hand_length);
		observeHandTrajectory(param.nTrajectory, param.minDist);
		createHandTrajectoryPC();	// optional 
	}
	if(!state.isHandDetected){
		state.handTrajectory.clear();
		state.gesture = NONE;
	}

    detectScroll();

    convertToXYZRGBPointCloud(rgbImage, depth_map, cloudRawPtr);

	return state;

}

void StateObserver::observeFace(float cameraAngle, float face_image_scale, float face_concent)
{
    state.face2d = faceDetection.detectFace(*rgbImage, face_image_scale);

	if(state.face2d.width != 0 && state.face2d.height != 0){
		state.isFaceDetected = true;		
		// face3D
		state.face3d.reset(new Cloud);
		float skinPercent = extractFace3D(cameraAngle, rgbImage, depth_map, state.face2d, state.face3d, state.facePos3d, face_concent);
		if(skinPercent < 0.5)
			state.isFaceDetected = false;
	}	
	else{
		state.isFaceDetected = false;	
	}
}

void StateObserver::observeHeadHand(float cameraAngle, float head_radius, float arm_length, int pointstep)
{	
	if(!state.isHeadDetected && !state.isFaceDetected){
		return;
	}
	else if(!state.isHeadDetected && state.isFaceDetected){
		state.isHeadDetected = true;
		state.headPos3d = state.facePos3d;
	}
	else if(state.isHeadDetected && state.isFaceDetected){
		state.isHeadDetected = true;
		state.headPos3d = state.facePos3d;
	}
	else if(state.isHeadDetected && !state.isFaceDetected){
		state.isHeadDetected = true;	
	}

	if(state.isHeadDetected){
		state.head3d.reset(new Cloud);
		state.hand3d.reset(new Cloud);
		extractHeadHand3D(cameraAngle, rgbImage, depth_map, head_radius, arm_length, pointstep);		
        if(state.hand3d->points.size() > 0)
			state.isHandDetected = true;
		else
			state.isHandDetected = false;
	}
	else{
		state.isHandDetected = false;
	}

}

float StateObserver::extractFace3D(float cameraAngle, const boost::shared_ptr<cv::Mat> &image, const XnDepthPixel* depth_map, cv::Rect face, CloudPtr &cloud, Eigen::Vector3d &pos, float concent)
{
	// resize face
	face.x = face.x + face.width *(0.5-concent/2.);
	face.y = face.y + face.height *(0.5-concent/2.);
	face.width = face.width * concent;
	face.height = face.height * concent;

	// initialize cloud;
	cloud->header.frame_id = rgb_frame_id_;
	cloud->height = face.height;
	cloud->width = face.width;
	cloud->is_dense = false;
	cloud->points.resize (cloud->height * cloud->width);

	pos[0] = 0.;
	pos[1] = 0.;
	pos[2] = 0.;

	register int idx = 0;
	int size = 0;
	int skinColorSize = 0;
	for (int v = face.y; v < face.y+face.height; ++v)
	{
		for (register int u = face.x; u < face.x+face.width; ++u, ++idx)
		{
			if (depth_map[v*cloudRawPtr->width + u] == 0 ||
				depth_map[v*cloudRawPtr->width + u] == depth_max_)
			{
				// bad point
			}
			else{
				PointT& pt = cloud->points[idx];				
				pt.z = static_cast<float> (depth_map[v*cloudRawPtr->width + u]) * 0.001f;
				pt.x = static_cast<float> (u-centerX) * pt.z * XtoZ * 0.001f;
				pt.y = static_cast<float> (v-centerY) * pt.z * YtoZ * 0.001f;
				
				float newZ = pt.z*cos(cameraAngle/180.*PI) - pt.y*sin(cameraAngle/180.*PI);
				float newY = pt.z*sin(cameraAngle/180.*PI) + pt.y*cos(cameraAngle/180.*PI);

				pt.z = newZ;
				pt.y = newY;

				cv::Vec3b pixel = rgbImage->at<cv::Vec3b>(v, u);

				pt.r = pixel[2];
				pt.g = pixel[1];
				pt.b = pixel[0];
				pt.a = 1;

				pos[0] += pt.x;
				pos[1] += pt.y;
				pos[2] += pt.z;
				size ++;

				// skin color 
				if(SkinDetection::isSkin(pixel))	skinColorSize++;
			}			
		}
	}
	pos /= (float)size;
	return (float)skinColorSize / (float)size;
}


void StateObserver::extractHeadHand3D(float cameraAngle, const boost::shared_ptr<cv::Mat> &image, const XnDepthPixel* depth_map, float head_radius, float arm_length, int pointstep)
{	
	// initialize cloud;
	state.head3d->header.frame_id = rgb_frame_id_;
	state.head3d->is_dense = false;	
	state.hand3d->header.frame_id = rgb_frame_id_;
	state.hand3d->is_dense = false;
	
	Eigen::Vector3d newHeadPos;
	newHeadPos[0] = 0.;
	newHeadPos[1] = 0.;
	newHeadPos[2] = 0.;
	int headSize = 0;

	state.handPos3d[0] = 0.;
	state.handPos3d[1] = 0.;
	state.handPos3d[2] = 0.;
	int handSize = 0;	

	Eigen::Vector3d st_hand;
	st_hand[0] = state.headPos3d[0];
	st_hand[1] = state.headPos3d[1] + head_radius*2;
	st_hand[2] = state.headPos3d[2] - head_radius;
	
	register int idx = 0;
	for (int v = 0; v < cloudRawPtr->height; v+=pointstep)
	{
		for (register int u = 0; u < cloudRawPtr->width; u+=pointstep, ++idx)
		{
			if (depth_map[v*cloudRawPtr->width + u] == 0 ||
				depth_map[v*cloudRawPtr->width + u] == depth_max_)
			{
				// bad point
			}
			else{
				PointT point;
				point.z = static_cast<float> (depth_map[v*cloudRawPtr->width + u]) * 0.001f;
				point.x = static_cast<float> (u-centerX) * point.z * XtoZ * 0.001f;
				point.y = static_cast<float> (v-centerY) * point.z * YtoZ * 0.001f;	

				float newZ = point.z*cos(cameraAngle/180.*PI) - point.y*sin(cameraAngle/180.*PI);
				float newY = point.z*sin(cameraAngle/180.*PI) + point.y*cos(cameraAngle/180.*PI);

				point.z = newZ;
				point.y = newY;
				
				// skin color
				cv::Vec3b pixel = rgbImage->at<cv::Vec3b>(v, u);					
				if(SkinDetection::isSkin(pixel)){
					point.r = pixel[2];
					point.g = pixel[1];
					point.b = pixel[0];
					point.a = 1;
					// head point
					double dist_head = sqrt(pow(point.x-state.headPos3d[0],2) + pow(point.y-state.headPos3d[1],2) + pow(point.z-state.headPos3d[2],2));
					if(dist_head <=head_radius){						
						state.head3d->points.push_back(point);
						newHeadPos[0] += point.x;
						newHeadPos[1] += point.y;
						newHeadPos[2] += point.z;
						headSize++;
					}

					// hand point
					double dist_hand = sqrt(pow(point.x-st_hand[0],2) + pow(point.y-st_hand[1],2) + pow(point.z-st_hand[2],2));
					if(point.y < st_hand[1] && point.z < st_hand[2] && dist_hand <= arm_length){
						state.hand3d->points.push_back(point);
						state.handPos3d[0] += point.x;
						state.handPos3d[1] += point.y;
						state.handPos3d[2] += point.z;
						handSize++;
					}
				}
			}			
		}
	}

	state.handPos3d /= (float)handSize;
	newHeadPos /= (float)headSize;
	state.headPos3d = newHeadPos;

}

void StateObserver::handPCFiltering(float hand_length, int minHandPoints)
{
	// mean: state.handPos3d;
	// calculate variance for each point
	state.hand3d_filtered.reset(new Cloud);
	state.hand3d_filtered->header.frame_id = rgb_frame_id_;
	state.hand3d_filtered->is_dense = false;	

	state.handPos3d_filtered[0] = 0.;
	state.handPos3d_filtered[1] = 0.;
	state.handPos3d_filtered[2] = 0.;
	int size_filtered = 0;
	int size = state.hand3d->points.size();
	
	// find the far most point	
	float maxDist;
	bool isFirstDetected = false;
	for(int i=0;i<size;i++){
		PointT point = state.hand3d->points.at(i);
		float dist = sqrt(pow(point.x-state.handPos3d[0],2)+pow(point.y-state.handPos3d[1],2)+pow(point.z-state.handPos3d[2],2));
		if(!isFirstDetected && point.y < state.handPos3d[1]){
			maxDist = dist;
			state.handPos3d_farMost[0] = point.x;
			state.handPos3d_farMost[1] = point.y;
			state.handPos3d_farMost[2] = point.z;
			isFirstDetected = true;
		}
		if(isFirstDetected && dist > maxDist && point.y < state.handPos3d[1]){
			maxDist = dist;
			state.handPos3d_farMost[0] = point.x;
			state.handPos3d_farMost[1] = point.y;
			state.handPos3d_farMost[2] = point.z;
		}
	}
	for(int i=0;i<state.hand3d->points.size();i++){
		PointT point = state.hand3d->points.at(i);
		float dist = sqrt(pow(point.x-state.handPos3d_farMost[0],2)+pow(point.y-state.handPos3d_farMost[1],2)+pow(point.z-state.handPos3d_farMost[2],2));
		if(dist < hand_length){
			size_filtered ++;
			state.hand3d_filtered->points.push_back(point);
			state.handPos3d_filtered[0] += point.x;
			state.handPos3d_filtered[1] += point.y;
			state.handPos3d_filtered[2] += point.z;
		}
	}
	state.handPos3d_filtered /= (float)size_filtered;	

    if(state.hand3d_filtered->points.size() > minHandPoints)
        state.isHandDetected = true;
    else
        state.isHandDetected = false;
}

void StateObserver::convertToXYZRGBPointCloud (const boost::shared_ptr<cv::Mat> &image,	const XnDepthPixel* depth_map, CloudPtr &cloud)
{	  	
	register int depth_idx = 0;

	for (int v = 0; v < cloud->height; ++v)
	{
		for (register int u = 0; u < cloud->width; ++u, ++depth_idx)
		{
			PointT& pt = cloud->points[depth_idx];
			/// @todo Different values for these cases
			// Check for invalid measurements
			if (depth_map[depth_idx] == 0 ||
				depth_map[depth_idx] == depth_max_)
			{
				pt.x = pt.y = pt.z = bad_point;
			}
			else
			{
				pt.z = static_cast<float> (depth_map[depth_idx]) * 0.001f;
				pt.x = static_cast<float> (u-centerX) * pt.z * XtoZ * 0.001f;
                pt.y = static_cast<float> (v-centerY) * pt.z * YtoZ * 0.001f;
			}
			cv::Vec3b pixel = rgbImage->at<cv::Vec3b>(v, u);
			pt.r = pixel[2];
			pt.g = pixel[1];
			pt.b = pixel[0];
		}
	}
}

void StateObserver::observeHandState(float hand_length)
{
	float distMeanFMP = (state.handPos3d_filtered-state.handPos3d_farMost).norm();
	if(distMeanFMP > hand_length / 2)
		state.handState = PALM;
	else
		state.handState = FIST;
}

void StateObserver::observeHandTrajectory(int nTrajectory, float minDist)
{
	if(state.handState == FIST){
//		if(state.timeover == false){
			if(state.handTrajectory.size() < nTrajectory)
				state.handTrajectory.push_back(state.handPos3d_filtered);
			else{
				detectGesture(minDist);
//				state.timeover = true;
				//for(int i=0;i<nTrajectory-1;i++)
				//	state.handTrajectory.at(i) = state.handTrajectory.at(i+1);
				//state.handTrajectory.at(nTrajectory-1) = state.handPos3d_filtered;
			}
//		}
//		else if(state.timeover)	state.gesture = NONE;
	}	
	if(state.handState == PALM){		
//		if(!state.timeover){
            if(state.handTrajectory.size() != 0){
                detectGesture(minDist);
            }
            else state.gesture = NONE;
//		}
//		state.timeover = false;
		state.handTrajectory.clear();
	}
}

void StateObserver::createHandTrajectoryPC()
{
	for(int i=0;i<state.handTrajectory.size();i++)
	{
		PointT point;
		Eigen::Vector3d pos = state.handTrajectory.at(i);
		point.x = pos[0];
		point.y = pos[1];
		point.z = pos[2];
		point.r = 0;
		point.g = 255;
		point.b = 0;
		point.a = 1;
		state.hand3d_filtered->points.push_back(point);
	}
}

void StateObserver::detectGesture(float minDist)
{
	Eigen::Vector3d start, end;
	start = state.handTrajectory.at(0);
	end = state.handTrajectory.at(state.handTrajectory.size()-1);
	if((start-end).norm() < minDist)
		state.gesture = NONE;
	else{
		float diffX = end[0] - start[0];
		float diffY = end[1] - start[1];
		float diffZ = end[2] - start[2];

		if(fabs(diffX) > fabs(diffY) && fabs(diffX) > fabs(diffZ)){
			if(diffX < 0)	state.gesture = RIGHT;			
			else			state.gesture = LEFT;
			
		}
		else if(fabs(diffY) > fabs(diffX) && fabs(diffY) > fabs(diffZ)){
			if(diffY < 0)	state.gesture = UP;
			else			state.gesture = DOWN;

		}
		else if(fabs(diffZ) > fabs(diffX) && fabs(diffZ) > fabs(diffY)){
			if(diffZ < 0)	state.gesture = PUSH;
			else			state.gesture = PULL;
		}
	}
	
}

void StateObserver::detectScroll()
{
    static float initY = 0.;
    if(!state.isHandDetected){
        state.scroll = 0;
        return;
    }
    else{
        if(state.handState == PALM){
            state.scroll = 0;
            initY = state.handPos3d_filtered[1];
        }
        if(state.handState == FIST){
            state.scroll = initY - state.handPos3d_filtered[1];
        }
    }
}
