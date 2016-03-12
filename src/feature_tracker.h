#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>

#include "nvx_feature_tracker.hpp"
#include <NVXIO/Application.hpp>
#include <NVXIO/FrameSource.hpp>
#include <NVXIO/Render.hpp>
#include <NVXIO/Utility.hpp>
#include <NVXIO/ConfigParser.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h> 
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#define MIN_DIST 30
#define MAX_CNT 100
#define FREQ 3
#define RANSAC true

const int NUM_OF_CAM = 1;
const int ROW = 480;
const int COL = 752;

using namespace std;
using namespace camodocal;
using namespace Eigen;

class FeatureTracker
{
	public:
		FeatureTracker();
		void changeType(vx_array vx_pt, vector<cv::Point2f> &cv_pt );
		void addFeatures();
		void printresult();
		void printvector(vector<cv::Point2f> &v);
		vector<cv::Point2f> undistortedPoints(std::vector<cv::Point2f> v);
		void ransac(std::vector<cv::Point2f> prev, std::vector<cv::Point2f> curr);

		int cnt = 0;
		double ransac_thres;

		camodocal::CameraPtr m_camera;

		nvx::FeatureTracker *tracker;
		//std::unique_ptr<nvx::FeatureTracker> tracker(nvx::FeatureTracker::createHarrisPyrLK(context, params));
		//std::unique_ptr<nvxio::Render> renderer(nvxio::createDefaultRender(context, "Feature Tracker", 752, 480));
		vx_image mask = NULL;
		bool isInit = false;

		vx_image src1;
		int id_count = 0;
		        
		double proc_ms = 0;
		vector<cv::Point2f> prev_pts, cur_pts, forw_pts, harris_pts, ransac_pts;
		vector<int> cur_ids;
		vector<int> prev_ids;
		vector<int> ransac_ids;
		vector<int> cur_track_cnt;
		vector<int> prev_track_cnt;
		vector<int> ransac_track_cnt;

};