#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
} // namespace backward


#include "feature_tracker.h"

using namespace std;
using namespace camodocal;



ros::Publisher pub_img;
FeatureTracker trackerData[2];
nvx::Timer totalTimer;
nvx::Timer trackTimer;
nvxio::ContextGuard context;
int NUM_OF_CAM; 
bool SHOW_IMAGE;
bool PUB_UV;



void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    totalTimer.tic();
    cout<<"img_callback"<<endl;
    cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    for(int i = 0 ;i < NUM_OF_CAM ;i++)
    {
        cv::Mat img;
        img = bridge_ptr->image.colRange(COL * i, COL * (i + 1));
        trackerData[i].image = img.clone();
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
         for(unsigned int j = 0; j < trackerData[i].prev_pts.size(); j++)
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
        ROS_INFO("Track Time %f",track_ms);
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
            for(unsigned int j = 0; j < trackerData[i].forw_pts.size(); j++)
            {
                if(trackerData[i].forw_pts[j].x != -1)
                {
                    trackerData[i].cur_pts.push_back(trackerData[i].forw_pts[j]);
                    trackerData[i].cur_ids.push_back(trackerData[i].prev_ids[j]);
                    trackerData[i].cur_track_cnt.push_back(++trackerData[i].prev_track_cnt[j]);

                    trackerData[i].ransac_pts.push_back(trackerData[i].prev_pts[j]);
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
            trackerData[i].tracker->optIn(trackerData[i].cur_pts); 
        }
    }

}

    //show and pub
if(trackerData[0].isInit && trackerData[0].cnt == 0)
{

    ROS_INFO("pub_image");
    sensor_msgs::PointCloud feature;
    sensor_msgs::ChannelFloat32 id_of_point;
    feature.header = img_msg->header;
    for(int i = 0 ;i < NUM_OF_CAM; i++)
    {
        auto un_pts = trackerData[i].undistortedPoints(trackerData[i].cur_pts);
        auto &ids = trackerData[i].cur_ids;
        trackerData[i].goodfeature.clear();
        for (unsigned int j = 0; j < ids.size(); j++)
        {
            int p_id = ids[j];
            geometry_msgs::Point32 p;
            p.x = un_pts[j].x;
            p.y = un_pts[j].y;
            p.z = 1;
            geometry_msgs::Point32 uv;
            uv.x = trackerData[i].cur_pts[j].x;
            uv.y = trackerData[i].cur_pts[j].y;
            uv.z = 1;

            if(uv.y < 10 || uv.y > ROW - 10)
            {
                trackerData[i].goodfeature.push_back(false);
                continue;
            }
            if(p.x!=p.x || p.y!=p.y)
            {
                ROS_WARN("Nan problem!");
                trackerData[i].goodfeature.push_back(false);
                continue;
            }

            if(p.x > 20 || p.y > 20)
            {
                trackerData[i].goodfeature.push_back(false);
                continue;
            }
            trackerData[i].goodfeature.push_back(true);
            if(PUB_UV)
                feature.points.push_back(uv);
            else
                feature.points.push_back(p);
            id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
        }
    }
    feature.channels.push_back(id_of_point);
    pub_img.publish(feature);

    if(SHOW_IMAGE)
    {

        ROS_INFO("Show image"); 

        nvx::Timer showTimer;
        showTimer.tic();    

        std::vector<cv::Mat> tmp_img;
        for(int i = 0 ;i < NUM_OF_CAM; i++)
        {
            cv::Mat color_img;
            cv::cvtColor(trackerData[i].image, color_img, CV_GRAY2RGB);
            tmp_img.push_back(color_img);
            for(unsigned j = 0; j < trackerData[i].cur_pts.size(); j++)
            {
                if(trackerData[i].goodfeature[j])
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].cur_track_cnt[j] / 20);
                    cv::circle(tmp_img[i], trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }
                else
                {
                    //cv::circle(tmp_img, cur_pts[i], 2, cv::Scalar(255, 255, 255 ), 2);
                }
            }
        }
//combine NUM_OF_CAMERA PICTURE
        cv::Size size(COL * NUM_OF_CAM, ROW); 
        cv::Mat img_merge;  


        img_merge.create(size, CV_8UC3);  

        for(int i = 0 ;i < NUM_OF_CAM; i++)
        {
            cv::Mat block_img = img_merge(cv::Rect(i * COL, 0, COL, ROW));
            tmp_img[i].copyTo(block_img);
        }


        cv::imshow("features",img_merge);
        double show_ms = showTimer.toc();
        ROS_INFO("Show Time %f",show_ms);
        //cout<<"debug show time "<<show_ms<<endl;

        cv::waitKey(1);


        }

    }


    //release
    for(int i = 0;i < NUM_OF_CAM; i++)
    {
        trackerData[i].cnt = (trackerData[i].cnt + 1) % FREQ;
        vxReleaseImage(&trackerData[i].src1);
    }

    double total_ms = totalTimer.toc();
    ROS_INFO("Total Time %f",total_ms);
    cout<<endl<<endl;
    ROS_WARN_COND(total_ms > 40, "processing over 40 ms");
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
    n.getParam("NUM_OF_CAM", NUM_OF_CAM);
    n.getParam("SHOW_IMAGE", SHOW_IMAGE);
    n.getParam("PUB_UV", PUB_UV);


    cout<<"ransac_thres    "<<ransac_thres<<endl;
    cout<<"NUM_OF_CAM      "<<NUM_OF_CAM<<endl;
    cout<<"pub uv   "<<PUB_UV<<endl;


    string calib_file[2];
    //n.getParam("calib_file", calib_file);
    for(int i = 0 ;i < NUM_OF_CAM; i++)
    {
        n.getParam("calib_file" + to_string(i), calib_file[i]);
        cout<<"came  "<<i <<"      calib_file    "<<calib_file[i] <<endl;

    }

    //cout<<"calib_file"<<calib_file<<endl;


    nvx::FeatureTracker::HarrisPyrLKParams params;

    params.pyr_levels = pyr_levels;
    params.lk_num_iters = lk_num_iters;
    params.lk_win_size = lk_win_size;
    params.harris_k = harris_k;
    params.harris_thresh = harris_thresh;
    params.harris_cell_size = harris_cell_size;
    params.array_capacity = array_capacity;


    // load mask
    cv::Mat mask_image;
    mask_image = cv::imread("/home/ubuntu/catkin_ws/src/feature_tracker_visionworks/config/mask.jpg", 0);

    vx_image mask;
    vx_imagepatch_addressing_t src1_addr;
    src1_addr.dim_x = mask_image.cols;
    src1_addr.dim_y = mask_image.rows;
    src1_addr.stride_x = sizeof(vx_uint8);
    src1_addr.stride_y = mask_image.step;
    void *src1_ptrs[] = {
        mask_image.data
    };

    mask = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &src1_addr, src1_ptrs, VX_IMPORT_TYPE_HOST);
    NVXIO_CHECK_REFERENCE(mask);


    for(int i = 0 ;i < NUM_OF_CAM; i++)
    {
        trackerData[i].m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        trackerData[i].tracker = nvx::FeatureTracker::createHarrisPyrLK(context, params);
        trackerData[i].ransac_thres = ransac_thres;
        trackerData[i].mask = mask;
    }

    ros::Subscriber sub_img = n.subscribe("image_raw", 10, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("image",1000);



    ros::spin();


}









