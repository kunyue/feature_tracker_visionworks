#include "feature_tracker.h"

using namespace std;
using namespace camodocal;



ros::Publisher pub_img;
FeatureTracker trackerData[NUM_OF_CAM];
nvx::Timer totalTimer;
nvx::Timer trackTimer;
nvxio::ContextGuard context;



void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    totalTimer.tic();
    cout<<"img_callback"<<endl;
    cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    for(int i = 0 ;i < NUM_OF_CAM ;i++)
    {
        cv::Mat img;
        img = bridge_ptr->image.rowRange(ROW * i, ROW * (i + 1));
        vx_imagepatch_addressing_t src1_addr;
        src1_addr.dim_x = img.cols;
        src1_addr.dim_y = img.rows;
        src1_addr.stride_x = sizeof(vx_uint8);
        src1_addr.stride_y = img.step;
        void *src1_ptrs[] = {
            img.data
        };

        trackerData[i].src1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &src1_addr, src1_ptrs, VX_IMPORT_TYPE_HOST);
        NVXIO_CHECK_REFERENCE(trackerData[i].src1);
    

        if( !trackerData[i].isInit)
        {
             trackerData[i].tracker->init( trackerData[i].src1,  trackerData[i].mask);
             trackerData[i].isInit = true;
             cout<<"isInit"<<endl;
        
             trackerData[i].changeType( trackerData[i].tracker->getHarrisFeatures() , trackerData[i].prev_pts);
            //printvector(prev_pts);
             trackerData[i].tracker->optIn( trackerData[i].prev_pts);
            for(unsigned int i = 0; i< trackerData[i].prev_pts.size(); i++)
            {

                 trackerData[i].prev_ids.push_back( trackerData[i].id_count);
                 trackerData[i].prev_track_cnt.push_back(1);
                 trackerData[i].id_count++;
            //cout<<"forw_pts_i  "<<i<<" x "<<prev_pts[i].x<<" y "<<prev_pts[i].y<<endl;
            }
        }
        else 
        {
		
            trackTimer.tic();  

		    ROS_INFO("Tracking");
            trackerData[i].tracker->track(trackerData[i].src1, trackerData[i].mask);

            double track_ms = trackTimer.toc();
            std::cout << "track Time : " << track_ms << " ms" << std::endl  ;
            if(trackerData[i].cnt !=0 )
            {
                ROS_INFO("Continue tracking");
                trackerData[i].changeType(trackerData[i].tracker->getOpticalFeatures() ,trackerData[i].forw_pts);
                trackerData[i].tracker->optIn(trackerData[i].forw_pts);  

            }
            else
            {
                ROS_INFO("processing tracking result");
                trackerData[i].changeType(trackerData[i].tracker->getOpticalFeatures() ,trackerData[i].forw_pts);
            //printvector(forw_pts);
                trackerData[i].cur_pts.clear();
                trackerData[i].cur_ids.clear();
                trackerData[i].cur_track_cnt.clear();
                trackerData[i].ransac_pts.clear();
                for(unsigned int i = 0; i < trackerData[i].forw_pts.size(); i++)
                {
                    if(trackerData[i].forw_pts[i].x != -1)
                    {
                        trackerData[i].cur_pts.push_back(trackerData[i].forw_pts[i]);
                        trackerData[i].cur_ids.push_back(trackerData[i].prev_ids[i]);
                        trackerData[i].cur_track_cnt.push_back(++trackerData[i].prev_track_cnt[i]);

                        trackerData[i].ransac_pts.push_back(trackerData[i].prev_pts[i]);
                    }
                }

                //ransac begin
                trackerData[i].ransac_ids = trackerData[i].cur_ids;
                trackerData[i].ransac_track_cnt = trackerData[i].cur_track_cnt;
                if(RANSAC)
                {
                    ROS_INFO("ransac begin");
                    trackerData[i].ransac(trackerData[i].ransac_pts, trackerData[i].cur_pts);
                }

                ROS_INFO("Add new features");
                trackerData[i].changeType(trackerData[i].tracker->getHarrisFeatures(), trackerData[i].harris_pts);
                ROS_INFO("Find harris features %d",(int)trackerData[i].harris_pts.size());
                //printvector(harris_pts);
                trackerData[i].addFeatures();
                ROS_INFO("Use features %d" ,(int)trackerData[i].cur_pts.size());
                //printresult();
                trackerData[i].prev_pts = trackerData[i].cur_pts;
                trackerData[i].prev_ids = trackerData[i].cur_ids;
                trackerData[i].prev_track_cnt = trackerData[i].cur_track_cnt;

                //ROS_BREAK();
                trackerData[i].tracker->optIn(trackerData[i].cur_pts);  
                //tracker->printPerfs();
                /*
                ROS_INFO("pub_image");
                sensor_msgs::PointCloud feature;
                sensor_msgs::ChannelFloat32 id_of_point;
                feature.header = img_msg->header;
                auto un_pts = undistortedPoints(cur_pts);
                auto &ids = cur_ids;
                std::vector<bool> goodfeature;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    int p_id = ids[j];
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    if(p.x!=p.x || p.y!=p.y)
                    {
                        ROS_WARN("Nan problem!");
                        goodfeature.push_back(false);
                        //ROS_BREAK();
                        continue;
                    }

                    if(p.x > 20 || p.y > 20)
                    {
                        //ROS_WARN("not good point");
                        continue;
                        goodfeature.push_back(false);

                    }
                    goodfeature.push_back(true);

                    feature.points.push_back(p);
                    id_of_point.values.push_back(p_id);
                }
                feature.channels.push_back(id_of_point);
                pub_img.publish(feature);
        
            */


            /*
            ROS_INFO("Show image"); 

            nvx::Timer showTimer;
            showTimer.tic();    

            cv::Mat tmp_img;
            cv::cvtColor(bridge_ptr->image, tmp_img, CV_GRAY2RGB);
            
            for(unsigned i = 0; i < cur_pts.size(); i++)
            {
                if(goodfeature[i])
                {
                    double len = std::min(1.0, 1.0 * cur_track_cnt[i] / 20);
                    cv::circle(tmp_img, cur_pts[i], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }
                else
                {
                    //cv::circle(tmp_img, cur_pts[i], 2, cv::Scalar(255, 255, 255 ), 2);
                }
            }
            cv::imshow("features",tmp_img);
            double show_ms = showTimer.toc();
            ROS_INFO("Total %d featrues",(int)cur_pts.size());
            //std::cout << "show Time : " << show_ms << " ms" << std::endl  ;

            
            cv::waitKey(1);
            */

            }

        }

        double total_ms = totalTimer.toc();
        std::cout << "Total Time : " << total_ms << " ms" << std::endl ;
        cout<<endl<<endl;
        ROS_WARN_COND(total_ms > 30, "processing over 30 ms");

        trackerData[i].cnt = (trackerData[i].cnt + 1) % FREQ;
        vxReleaseImage(&trackerData[i].src1);
    }
}



int main(int argc, char* argv[])
{

    //*************************************ros_init**************************
    ros::init(argc,argv,"feature");
    ros::NodeHandle n("~");
    double harris_k, harris_thresh,ransac_thres;
    int pyr_levels,lk_num_iters,lk_win_size,harris_cell_size,array_capacity;

    n.getParam("harris_k", harris_k);
    n.getParam("harris_thresh", harris_thresh);
    n.getParam("pyr_levels", pyr_levels);
    n.getParam("lk_num_iters", lk_num_iters);
    n.getParam("lk_win_size", lk_win_size);
    n.getParam("harris_cell_size", harris_cell_size);
    n.getParam("array_capacity", array_capacity);
    
    n.getParam("ransac_thresh", ransac_thres);

    cout<<"ransac_thres    "<<ransac_thres<<endl;


    string calib_file;
    n.getParam("calib_file", calib_file);
    cout<<"calib_file   "<<calib_file<<endl;


    nvx::FeatureTracker::HarrisPyrLKParams params;

    params.pyr_levels = pyr_levels;
    params.lk_num_iters = lk_num_iters;
    params.lk_win_size = lk_win_size;
    params.harris_k = harris_k;
    params.harris_thresh = harris_thresh;
    params.harris_cell_size = harris_cell_size;
    params.array_capacity = array_capacity;

    for(int i = 0 ;i < NUM_OF_CAM; i++)
    {
        trackerData[i].m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
        trackerData[i].tracker = nvx::FeatureTracker::createHarrisPyrLK(context, params);
        trackerData[i].ransac_thres = ransac_thres;
    }

    ros::Subscriber sub_img = n.subscribe("image_raw", 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("image",1000);
           
    ros::spin();


}
