#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.141592
#define SHOW_FPS 1
#if SHOW_FPS
#define FPS_CALC(_WHAT_) \
	do \
{ \
	static unsigned count = 0;\
	static double last = pcl::getTime ();\
	++count; \
	if (pcl::getTime() - last >= 1.0) \
	{ \
	double now = pcl::getTime (); \
	std::cout << "Average framerate("<< _WHAT_ << "): " << double(count)/double(now - last) << " Hz" <<  std::endl; \
	count = 0; \
	last = now; \
	} \
}while(false)
#else
#define FPS_CALC(_WHAT_) \
	do \
{ \
}while(false)
#endif

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;

using namespace std;
