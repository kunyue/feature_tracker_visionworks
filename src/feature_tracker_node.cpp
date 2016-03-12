
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
#define ROW 480
#define COL 752
#define MAX_CNT 100
#define FREQ 3
#define RANSAC true
#define RANSAC_THRESsHOLD 3
//
// Utility functions
//
int cnt = 0;
double ransac_thres;
using namespace std;
using namespace camodocal;
camodocal::CameraPtr m_camera;
nvxio::ContextGuard context;
nvx::FeatureTracker::HarrisPyrLKParams params;
nvx::FeatureTracker *tracker;
//std::unique_ptr<nvx::FeatureTracker> tracker(nvx::FeatureTracker::createHarrisPyrLK(context, params));

//std::unique_ptr<nvxio::Render> renderer(nvxio::createDefaultRender(context, "Feature Tracker", 752, 480));
vx_image mask = NULL;
bool isInit = false;
nvx::Timer totalTimer;
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

ros::Publisher pub_img;
/*
static void displayState(nvxio::Render *renderer,
                         int width, int height,
                         double proc_ms, double total_ms)
{
    std::ostringstream txt;

    txt << std::fixed << std::setprecision(1);

    nvxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 255}, {10, 10}};

    txt << "Source size: " << width << 'x' << height << std::endl;
    txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
    txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;

    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";
    renderer->putText(txt.str(), style);
}

static void drawArrows(nvxio::Render *renderer, vx_array old_points, vx_array new_points)
{
    nvxio::Render::FeatureStyle featureStyle = { { 255, 0, 0, 255 }, 4.0f };
    nvxio::Render::LineStyle arrowStyle = {{0, 255, 0, 255}, 1};

    //renderer->putArrows(old_points, new_points, arrowStyle);
    renderer->putFeatures(old_points, featureStyle);
    //renderer->putFeatures(new_points, featureStyle);
}

*/

void changeType(vx_array vx_pt, vector<cv::Point2f> &cv_pt )
{

    vx_size vCount = 0;
    vxQueryArray(vx_pt, VX_ARRAY_ATTRIBUTE_NUMITEMS, &vCount, sizeof(vCount));
    vx_enum item_type = 0;
    NVXIO_SAFE_CALL( vxQueryArray(vx_pt, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type, sizeof(item_type)) );
    NVXIO_ASSERT( item_type == NVX_TYPE_POINT2F) ;
    
    vx_size stride;
    void * featureData = NULL;
    NVXIO_SAFE_CALL( vxAccessArrayRange(vx_pt, 0, vCount, &stride,
                                        (void**)&featureData, VX_READ_ONLY) );
    int id = 0;

    cv_pt.resize(vCount);
    for (vx_size i = 0; i < vCount; i++)
    {
        nvx_point2f_t feature = vxArrayItem(nvx_point2f_t, featureData, i, stride);
        cv_pt[id].x = feature.x;
        cv_pt[id].y = feature.y;
        id++;
     
    }
    NVXIO_SAFE_CALL( vxCommitArrayRange(vx_pt, 0, vCount, featureData) );
    cv_pt.resize(id);
    //cout<<"point size   "<<cv_pt.size()<<"  id  "<<id<<endl;
}

void addFeatures()
{
    double minDistance = MIN_DIST;

    const int cell_size = minDistance;
    const int grid_width = (COL + cell_size -1) / cell_size;
    const int grid_height = (ROW + cell_size -1) / cell_size;
    minDistance *= minDistance;

    vector<vector<cv::Point2f>> grid(grid_width * grid_height);
    vector<cv::Point2f> new_cur_pts;
    vector<int> new_cur_track_cnt;
    vector<int> new_cur_ids;
    int prev_num, now_num;
    prev_num = cur_pts.size();

    for(unsigned int i = 0; i < cur_pts.size(); i++)
    {

        cv::Point2f p = cv::Point2f(cur_pts[i].x,cur_pts[i].y);
        bool good = true;
        int x_cell = p.x / cell_size;
        int y_cell = p.y / cell_size;
        int x1 = x_cell - 1;
        int y1 = y_cell - 1;
        int x2 = x_cell + 1;
        int y2 = y_cell + 1;

        x1 = std::max(0,x1);
        y1 = std::max(0,y1);
        x2 = std::min(grid_width -1 , x2);
        y2 = std::min(grid_height-1,y2);

        for(int yy = y1; yy <= y2; yy++)
            for(int xx = x1; xx<=x2; xx++)
                for(auto &pp : grid[yy * grid_width + xx])
                {
                    float dx = p.x - pp.x;
                    float dy = p.y - pp.y;
                    if(dx * dx + dy * dy < minDistance)
                    {
                        good = false;
                        goto break_out1;
                    }
                }
        break_out1:
        if(good)
        {
            grid[y_cell * grid_width + x_cell].push_back(p);
            new_cur_pts.push_back(p);
            new_cur_ids.push_back(cur_ids[i]);
            new_cur_track_cnt.push_back(cur_track_cnt[i]);
        }

    }
    now_num = new_cur_pts.size();
    ROS_INFO("Delete %d features",prev_num - now_num);

    prev_num = now_num;

    for(auto &p : harris_pts)
    {
        if((int)new_cur_pts.size() >= MAX_CNT)
        {
            ROS_INFO("number of points > max_cnt, do not add new features");
            break;
        }
        bool good = true;
        int x_cell = p.x / cell_size;
        int y_cell = p.y / cell_size;
        int x1 = x_cell - 1;
        int y1 = y_cell - 1;
        int x2 = x_cell + 1;
        int y2 = y_cell + 1;

        x1 = std::max(0,x1);
        y1 = std::max(0,y1);
        x2 = std::min(grid_width -1 , x2);
        y2 = std::min(grid_height-1,y2);

        for(int yy = y1; yy <= y2; yy++)
            for(int xx = x1; xx<=x2; xx++)
                for(auto &pp : grid[yy * grid_width + xx])
                {
                    float dx = p.x - pp.x;
                    float dy = p.y - pp.y;
                    if(dx * dx + dy * dy < minDistance)
                    {
                        good = false;
                        goto break_out2;
                    }
                }
        break_out2:
        if(good)
        {
            grid[y_cell * grid_width + x_cell].push_back(p);
            new_cur_pts.push_back(p);
            new_cur_ids.push_back(id_count);
            id_count++;
            new_cur_track_cnt.push_back(1);
        }

    }

    now_num = new_cur_pts.size();
    ROS_INFO("add %d features",now_num - prev_num);

    cur_pts = new_cur_pts;
    cur_track_cnt = new_cur_track_cnt;
    cur_ids = new_cur_ids;

}
void printresult()
{
    for(unsigned int i = 0; i< cur_pts.size();i++)
        cout<<"id  "<<cur_ids[i]<<"  track_num  "<<cur_track_cnt[i]<<" points x "<<cur_pts[i].x<<" y "<<cur_pts[i].y<<endl;
}

void printvector(vector<cv::Point2f> &v)
{
    for(unsigned int i = 0; i < v.size(); i++)                           
        cout<<"id  "<<i<<"  x "<<v[i].x<<"  y "<<v[i].y<<endl;

    cout<<"print finish"<<endl;
}



vector<cv::Point2f> undistortedPoints(std::vector<cv::Point2f> v)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < v.size(); i++)
    {
        Eigen::Vector2d a(v[i].x, v[i].y);
        Eigen::Vector3d b,c;
        m_camera->liftProjective(a, b);
        //m_camera->liftSphere(a, c);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        if(b.z()!=b.z())
        {
            ROS_WARN_COND(b.z()!=b.z(),"nan false");
            //ROS_BREAK();
        }
        //cout<<"Projective   "<<"x  "<<b.x() <<"   y   "<<b.y()<<"   z   "<<b.z()<<endl;
        //cout<<"Sphere   "<<"x  "<<c.x()  <<"   y   "<<c.y() <<"   z   "<<c.z()<<endl;
        /*
        for(int i = 1 ; i < ROW ; i=i+4)
            for(int j = 1 ; j < COL ; j=j+4)
            {
                Eigen::Vector2d c(i,j);
                Eigen::Vector3d d;
                m_camera->liftProjective(c, d);
                cout<<"u  "<<i<<"   v   "<<j<<endl;
                cout<<"x  "<<d.x()  <<"   y   "<<d.y() <<"   z   "<<d.z()<<endl;
                cout<<"x  "<<d.x() / d.z() <<"   y   "<<d.y() / d.z()<<endl;
                if(d.z()!=d.z())
                {
                    cout<<"false"<<endl;
                }

            }
        ROS_BREAK();
        */
        

    }
    return un_pts;
}

void ransac(std::vector<cv::Point2f> prev, std::vector<cv::Point2f> curr)
{
    //ROS_INFO("ransac begin!");
    if((int)curr.size() < 4 )
        return;
    vector<uchar> status;
        // ransac after undistort
    std::vector<cv::Point2f> prev_un = undistortedPoints(prev);
    std::vector<cv::Point2f> curr_un = undistortedPoints(curr);

    cv::findFundamentalMat(prev_un, curr_un, cv::FM_RANSAC, ransac_thres, 0.99, status);
    int num = curr.size();
    cur_pts.clear();
    cur_ids.clear();
    cur_track_cnt.clear();


    for(unsigned int i = 0; i < curr.size(); i++)
    {
        //cout<<"prev_points "<<prev[i].x<<"  "<<prev[i].y<<"  curr points "<<curr[i].x<<"   "<<curr[i].y<<"status  "<<(int)status[i]<<endl;
        if((int)status[i] == 1)
        {
            cur_pts.push_back(curr[i]);
            cur_ids.push_back(ransac_ids[i]);
            cur_track_cnt.push_back(ransac_track_cnt[i]);
        }
    }

    ROS_INFO("RANSAC delete %d features" , num - (int)cur_pts.size());

}


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    totalTimer.tic();
    cout<<"img_callback"<<endl;
    cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    vx_imagepatch_addressing_t src1_addr;
    src1_addr.dim_x = bridge_ptr->image.cols;
    src1_addr.dim_y = bridge_ptr->image.rows;
    src1_addr.stride_x = sizeof(vx_uint8);
    src1_addr.stride_y = bridge_ptr->image.step;
    void *src1_ptrs[] = {
        bridge_ptr->image.data
    };

    src1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &src1_addr, src1_ptrs, VX_IMPORT_TYPE_HOST);
    NVXIO_CHECK_REFERENCE(src1);

    if(!isInit)
    {
        tracker->init(src1, mask);
        isInit = true;
        cout<<"isInit"<<endl;
        
        changeType(tracker->getHarrisFeatures() ,prev_pts);
        //printvector(prev_pts);
        tracker->optIn(prev_pts);
        for(unsigned int i = 0; i<prev_pts.size(); i++)
        {

            prev_ids.push_back(id_count);
            prev_track_cnt.push_back(1);
            id_count++;
            //cout<<"forw_pts_i  "<<i<<" x "<<prev_pts[i].x<<" y "<<prev_pts[i].y<<endl;
        }
    }
    else 
    {
		nvx::Timer trackTimer;
        trackTimer.tic();  

		ROS_INFO("Tracking");
        tracker->track(src1, mask);

        double track_ms = trackTimer.toc();
        std::cout << "track Time : " << track_ms << " ms" << std::endl  ;
        if(cnt !=0 )
        {
            ROS_INFO("Continue tracking");
            changeType(tracker->getOpticalFeatures() ,forw_pts);
            tracker->optIn(forw_pts);  

        }
        else
        {
            ROS_INFO("processing tracking result");
            changeType(tracker->getOpticalFeatures() ,forw_pts);
            //printvector(forw_pts);
            cur_pts.clear();
            cur_ids.clear();
            cur_track_cnt.clear();
            ransac_pts.clear();
            for(unsigned int i = 0; i < forw_pts.size(); i++)
            {
                if(forw_pts[i].x != -1)
                {
                    cur_pts.push_back(forw_pts[i]);
                    cur_ids.push_back(prev_ids[i]);
                    cur_track_cnt.push_back(++prev_track_cnt[i]);

                    ransac_pts.push_back(prev_pts[i]);
                }
            }

            //ransac begin
            ransac_ids = cur_ids;
            ransac_track_cnt = cur_track_cnt;
            if(RANSAC)
            {
                ROS_INFO("ransac begin");
                ransac(ransac_pts, cur_pts);
            }

            ROS_INFO("Add new features");
            changeType(tracker->getHarrisFeatures(), harris_pts);
            ROS_INFO("Find harris features %d",(int)harris_pts.size());
            //printvector(harris_pts);
            addFeatures();
            ROS_INFO("Use features %d" ,(int)cur_pts.size());
            //printresult();
            prev_pts = cur_pts;
            prev_ids = cur_ids;
            prev_track_cnt = cur_track_cnt;

            //ROS_BREAK();
            tracker->optIn(cur_pts);  
            //tracker->printPerfs();

            /*
            
            nvx::Timer renderTimer;
            renderTimer.tic();  

            renderer->putImage(src1);
            drawArrows(renderer.get(), tracker->getOptIn(), tracker->getOptIn());
            //displayState(renderer.get(), bridge_ptr->image.cols, bridge_ptr->image.rows, proc_ms, total_ms);
            renderer->flush();

            double render_ms = renderTimer.toc();
            std::cout << "render Time : " << render_ms << " ms" << std::endl  ;
            */



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

        }
    }

    double total_ms = totalTimer.toc();
    std::cout << "Total Time : " << total_ms << " ms" << std::endl ;
    cout<<endl<<endl;
    ROS_WARN_COND(total_ms > 30, "processing over 30 ms");

    cnt = (cnt + 1) % FREQ;
    vxReleaseImage(&src1);
}


struct EventData
{
    EventData(): shouldStop(false), pause(false) {}

    bool shouldStop;
    bool pause;
};

static void eventCallback(void* eventData, vx_char key, vx_uint32, vx_uint32)
{
    EventData* data = static_cast<EventData*>(eventData);

    if (key == 27)
    {
        data->shouldStop = true;
    }
    else if (key == 32)
    {
        data->pause = !data->pause;
    }
}



int main(int argc, char* argv[])
{

    //*************************************ros_init**************************
    ros::init(argc,argv,"feature");
    ros::NodeHandle n("~");
    double harris_k, harris_thresh;
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
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);


    params.pyr_levels = pyr_levels;
    params.lk_num_iters = lk_num_iters;
    params.lk_win_size = lk_win_size;
    params.harris_k = harris_k;
    params.harris_thresh = harris_thresh;
    params.harris_cell_size = harris_cell_size;
    params.array_capacity = array_capacity;


    tracker = nvx::FeatureTracker::createHarrisPyrLK(context, params);

    ros::Subscriber sub_img = n.subscribe("image_raw", 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("image",1000);

        
 /*   
        if (!renderer)
        {
            std::cerr << "Can't create a renderer" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        EventData eventData;
        renderer->setOnKeyboardEventCallback(eventCallback, &eventData);
*/
    

        vxRegisterLogCallback(context, &nvxio::stdoutLogCallback, vx_false_e);

    ros::spin();


}
