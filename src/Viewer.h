#pragma once

#include "common.h"
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/openni_camera/openni_image.h>
//#include "image_viewer_threaded.h"

class Viewer
{
public:
	Viewer(void);
	~Viewer(void);

	void showCloud(const CloudPtr &cloud);
	void showRGB (const boost::shared_ptr<openni_wrapper::Image> &image);
	
public:
	boost::shared_ptr<pcl::visualization::CloudViewer> viewer;
	//pcl::visualization::ImageViewerThreaded imgViewer;
	

private:
	vector<string> nameList;
	static boost::mutex updateModelMutex;
	CloudPtr cloudRawPtr;

};
