#include "Viewer.h"
#include <opencv/cv.h>

boost::mutex Viewer::updateModelMutex;

Viewer::Viewer(void)
{	
	viewer.reset(new pcl::visualization::CloudViewer("3D viewer"));	
}

Viewer::~Viewer(void)
{
}

void Viewer::showCloud(const CloudPtr &cloud)
{	
	viewer->showCloud(cloud);
}


void Viewer::showRGB (const boost::shared_ptr<openni_wrapper::Image> &image)
{
	unsigned int height = image->getHeight(); 
	unsigned int width = image->getWidth(); 
	cv::Mat frameBGR=cv::Mat(image->getHeight(),image->getWidth(),CV_8UC3); 

	image->fillRGB(frameBGR.cols,frameBGR.rows,frameBGR.data,frameBGR.step); 

	//-- 3. Apply the classifier to the frame 
	if( !frameBGR.empty() ) 
	{ 
		unsigned char *rgb_buffer; 
		rgb_buffer = (unsigned char *) malloc(sizeof (char)*(width*height*3)); 
		for(int j=0; j<height; j++) 
		{ 
			for(int i=0; i<width; i++) 
			{ 
				rgb_buffer[(j*width + i)*3+0] = frameBGR.at<cv::Vec3b>(j,i)[0];  // B 
				rgb_buffer[(j*width + i)*3+1] = frameBGR.at<cv::Vec3b>(j,i)[1];  // G 
				rgb_buffer[(j*width + i)*3+2] = frameBGR.at<cv::Vec3b>(j,i)[2];  // R 
				//std::cout << (j*width + i)*3+0 << "," << (j*width + i)*3+1 << "," << (j*width + i)*3+2 << "," << std::endl; 
			} 
		}	


		//		if(!imgViewer.wasStopped()) 
		//			imgViewer.showRGBImage(rgb_buffer,width,height); 

		delete rgb_buffer;
	} 
	else 
	{ printf(" --(!) No captured frame -- Break!"); } 
}