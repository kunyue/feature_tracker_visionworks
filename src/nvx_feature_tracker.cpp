
/*
# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "nvx_feature_tracker.hpp"

#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>

#include <VX/vxu.h>
#include <NVX/nvx.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <NVXIO/Utility.hpp>
#include "NVXIO/Application.hpp"
using namespace std;

namespace 
{
    //
    // FeatureTracker based on Harris feature + PyrLK optical flow
    //

    class FeatureTrackerHarrisPyrLK : public nvx::FeatureTracker
    {
    public:
        FeatureTrackerHarrisPyrLK(vx_context context, const HarrisPyrLKParams& params);
        ~FeatureTrackerHarrisPyrLK();

        void init(vx_image firstFrame, vx_image mask);
        void track(vx_image newFrame, vx_image mask);

        vx_array getPrevFeatures() const;
        vx_array getCurrFeatures() const;
        vx_array getOpticalFeatures() const;
        vx_array getHarrisFeatures() const;
        vx_array getOptIn() const;
        void optIn(std::vector<cv::Point2f> &v) ;

        void printPerfs() const;

    private:
        void createDataObjects();

        void processFirstFrame(vx_image frame, vx_image mask);
        void createMainGraph(vx_image frame, vx_image mask);

        void release();

        HarrisPyrLKParams params_;

        vx_context context_;

        // Format for current frames
        vx_df_image format_;
        vx_uint32 width_;
        vx_uint32 height_;

        // Pyramids for two successive frames
        vx_delay pyr_delay_;

        // Points to track for two successive frames
        vx_delay pts_delay_;

        // Tracked points
        vx_array kp_curr_list_;
        vx_array temp;

        vx_array opt_in;
        vx_array opt_feature;
        vx_array harris_feature;

        // Main graph
        vx_graph main_graph_;

        // Node from main graph (used to print performance results)
        vx_node cvt_color_node_;
        vx_node pyr_node_;
        vx_node opt_flow_node_;
        vx_node feature_track_node_;
    };

    FeatureTrackerHarrisPyrLK::FeatureTrackerHarrisPyrLK(vx_context context, const HarrisPyrLKParams& params) :
        params_(params)
    {
        context_ = context;

        format_ = VX_DF_IMAGE_VIRT;
        width_ = 0;
        height_ = 0;

        pyr_delay_ = 0;
        pts_delay_ = 0;
        kp_curr_list_ = 0;
        temp = 0;
        opt_in =0;
        opt_feature = 0;
        harris_feature = 0;

        main_graph_ = 0;
        cvt_color_node_ = 0;
        pyr_node_ = 0;
        opt_flow_node_ = 0;
        feature_track_node_ = 0;
    }

    FeatureTrackerHarrisPyrLK::~FeatureTrackerHarrisPyrLK()
    {
        release();
    }

    void FeatureTrackerHarrisPyrLK::init(vx_image firstFrame, vx_image mask)
    {
        // Check input format

        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        //NVXIO_ASSERT(format == VX_DF_IMAGE_RGBX);

        if (mask)
        {
            vx_df_image mask_format = VX_DF_IMAGE_VIRT;
            vx_uint32 mask_width = 0;
            vx_uint32 mask_height = 0;

            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_FORMAT, &mask_format, sizeof(mask_format)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

            NVXIO_ASSERT(mask_format == VX_DF_IMAGE_U8);
            NVXIO_ASSERT(mask_width == width);
            NVXIO_ASSERT(mask_height == height);
        }

        // Re-create graph if the input size was changed
        

        if (width != width_ || height != height_)
        {
            release();

            format_ = format;
            width_ = width;
            height_ = height;
           
            createDataObjects();

            createMainGraph(firstFrame, mask);
        }


        // Process first frame

        processFirstFrame(firstFrame, mask);
    }

    void FeatureTrackerHarrisPyrLK::track(vx_image newFrame, vx_image mask)
    {
        // Check input format

        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        //NVXIO_ASSERT(format == format_);
        NVXIO_ASSERT(width == width_);
        NVXIO_ASSERT(height == height_);

        if (mask)
        {
            vx_df_image mask_format = VX_DF_IMAGE_VIRT;
            vx_uint32 mask_width = 0;
            vx_uint32 mask_height = 0;

            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_FORMAT, &mask_format, sizeof(mask_format)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

            NVXIO_ASSERT(mask_format == VX_DF_IMAGE_U8);
            NVXIO_ASSERT(mask_width == width_);
            NVXIO_ASSERT(mask_height == height_);
        }
        //************************change*******************************************************
        // Update input parameters for next graph execution
        //NVXIO_SAFE_CALL( vxSetParameterByIndex(cvt_color_node_, 0, (vx_reference)newFrame) );
        NVXIO_SAFE_CALL( vxSetParameterByIndex(pyr_node_, 0, (vx_reference)newFrame) );
        //NVXIO_SAFE_CALL( vxSetParameterByIndex(feature_track_node_, 2, (vx_reference)mask) );
        NVXIO_SAFE_CALL( vxSetParameterByIndex(feature_track_node_, 0, (vx_reference)newFrame) );


        //**************************************************************************************
        // Age the delay objects (pyramid, points to track) before graph execution
        NVXIO_SAFE_CALL( vxAgeDelay(pyr_delay_) );
        NVXIO_SAFE_CALL( vxAgeDelay(pts_delay_) );

        // Process graph
        NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );
    }

    vx_array FeatureTrackerHarrisPyrLK::getPrevFeatures() const
    {
        return (vx_array)vxGetReferenceFromDelay(pts_delay_, -1);
    }

    vx_array FeatureTrackerHarrisPyrLK::getCurrFeatures() const
    {
        //return kp_curr_list_;

        return (vx_array)vxGetReferenceFromDelay(pts_delay_, 0);
    }

    vx_array FeatureTrackerHarrisPyrLK::getOpticalFeatures() const
    {
        return opt_feature;
    }

    vx_array FeatureTrackerHarrisPyrLK::getHarrisFeatures() const
    {
        return harris_feature;
    }


    vx_array FeatureTrackerHarrisPyrLK::getOptIn() const
    {
        return opt_in;
    }


    void FeatureTrackerHarrisPyrLK::optIn(std::vector<cv::Point2f> &v) 
    {
        if(v.size()==0)
        {
            cout<<"lost all points"<<endl;
        }
        vx_size vCount = 0;
        vxQueryArray(opt_in, VX_ARRAY_ATTRIBUTE_NUMITEMS, &vCount, sizeof(vCount));
        if((int)vCount >(int)v.size() )
        {
            vCount = v.size();
            NVXIO_SAFE_CALL( vxTruncateArray (opt_in, vCount) );
        }
        else if ((int)vCount < (int)v.size())
        {
            nvx_point2f_t add;
            void * ptr = &add;
            vx_size num_add = v.size() - vCount;
            NVXIO_SAFE_CALL( vxAddArrayItems (opt_in, num_add, ptr ,0));
        }
        vx_size stride;
        void * featureData = NULL;
        vCount = v.size();
        NVXIO_SAFE_CALL( vxAccessArrayRange(opt_in, 0, vCount, &stride, &featureData, VX_READ_AND_WRITE) );
        for (vx_size i = 0; i < vCount; i++)
        {
            vxArrayItem(nvx_point2f_t, featureData, i, stride).x = v[i].x;
            vxArrayItem(nvx_point2f_t, featureData, i, stride).y = v[i].y;
        }
        NVXIO_SAFE_CALL( vxCommitArrayRange(opt_in, 0, vCount, featureData) );
    }


    void FeatureTrackerHarrisPyrLK::printPerfs() const
    {
        vx_size num_items = 0;
        NVXIO_SAFE_CALL( vxQueryArray((vx_array)vxGetReferenceFromDelay(pts_delay_, -1), VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)) );
        //std::cout << "Found " << num_items << " Features" << std::endl;

        vx_perf_t perf;

        NVXIO_SAFE_CALL( vxQueryGraph(main_graph_, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        //std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

        //NVXIO_SAFE_CALL( vxQueryNode(cvt_color_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        //std::cout << "\t Color Convert Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

        NVXIO_SAFE_CALL( vxQueryNode(pyr_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        //std::cout << "\t Pyramid Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

        //NVXIO_SAFE_CALL( vxQueryNode(feature_track_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        //std::cout << "\t Feature Track Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

        NVXIO_SAFE_CALL( vxQueryNode(opt_flow_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        //std::cout << "\t Optical Flow Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;
    }

    void FeatureTrackerHarrisPyrLK::release()
    {
        format_ = VX_DF_IMAGE_VIRT;
        width_ = 0;
        height_ = 0;

        vxReleaseDelay(&pyr_delay_);
        vxReleaseDelay(&pts_delay_);
        vxReleaseArray(&kp_curr_list_);
        vxReleaseArray(&temp);
        vxReleaseArray(&opt_in);
        vxReleaseArray(&harris_feature);
        vxReleaseArray(&opt_feature);


        vxReleaseNode(&cvt_color_node_);
        vxReleaseNode(&pyr_node_);
        vxReleaseNode(&opt_flow_node_);
        vxReleaseNode(&feature_track_node_);

        vxReleaseGraph(&main_graph_);
    }

    // This function creates data objects that are not entirely linked to graphs
    void FeatureTrackerHarrisPyrLK::createDataObjects()
    {
        // Image pyramids for two successive frames are necessary for the computation.
        // A delay object with 2 slots is created for this purpose
        vx_pyramid pyr_exemplar = vxCreatePyramid(context_, params_.pyr_levels, VX_SCALE_PYRAMID_HALF, width_, height_, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(pyr_exemplar);
        pyr_delay_ = vxCreateDelay(context_, (vx_reference)pyr_exemplar, 2);
        NVXIO_CHECK_REFERENCE(pyr_delay_);
        vxReleasePyramid(&pyr_exemplar);

        // Input points to track need to kept for two successive frames.
        // A delay object with 2 slots is created for this purpose
        vx_array pts_exemplar = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(pts_exemplar);
        pts_delay_ = vxCreateDelay(context_, (vx_reference)pts_exemplar, 2);
        NVXIO_CHECK_REFERENCE(pts_delay_);
        vxReleaseArray(&pts_exemplar);

        // Create the list of tracked points. This is the output of the frame processing
        kp_curr_list_ = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(kp_curr_list_);
        temp = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(temp);
        opt_in = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(opt_in);       
        opt_feature = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(opt_feature);        
        harris_feature = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(harris_feature);
    }

    //
    // See feature_tracker_user_guide.md for explanation
    //
    void FeatureTrackerHarrisPyrLK::processFirstFrame(vx_image frame, vx_image mask)
    {

        //vx_image frameGray = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_U8);
        vx_image frameGray = frame;
        NVXIO_CHECK_REFERENCE(frameGray);

        //NVXIO_SAFE_CALL( vxuColorConvert(context_, frame, frameGray) );
        NVXIO_SAFE_CALL( vxuGaussianPyramid(context_, frameGray, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0)) );
        NVXIO_SAFE_CALL( nvxuHarrisTrack(context_, frameGray, harris_feature, mask, 0,
                                         params_.harris_k, params_.harris_thresh, params_.harris_cell_size, NULL) );
        //NVXIO_SAFE_CALL( nvxuHarrisTrack(context_, frameGray, (vx_array)vxGetReferenceFromDelay(pts_delay_, 0), mask, 0,
        //                                 params_.harris_k, params_.harris_thresh, params_.harris_cell_size, NULL) );

        vxReleaseImage(&frameGray);
    }

    //
    // See feature_tracker_user_guide.md for explanation
    //
    void FeatureTrackerHarrisPyrLK::createMainGraph(vx_image frame, vx_image mask)
    {
        main_graph_ = vxCreateGraph(context_);
        NVXIO_CHECK_REFERENCE(main_graph_);

        // Intermediate images. Both images are created as 'virtual' in order to inform the OpenVX
        // framework that the application will never access their content.
        //vx_image frameGray = vxCreateVirtualImage(main_graph_, width_, height_, VX_DF_IMAGE_U8);
        vx_image frameGray = frame;
        NVXIO_CHECK_REFERENCE(frameGray);

        // Lucas-Kanade optical flow node
        // Note: keypoints of the previous frame are also given as 'new points estimates'
        vx_float32 lk_epsilon = 0.01f;
        vx_scalar s_lk_epsilon = vxCreateScalar(context_, VX_TYPE_FLOAT32, &lk_epsilon);
        NVXIO_CHECK_REFERENCE(s_lk_epsilon);

        vx_scalar s_lk_num_iters = vxCreateScalar(context_, VX_TYPE_UINT32, &params_.lk_num_iters);
        NVXIO_CHECK_REFERENCE(s_lk_num_iters);

        vx_bool lk_use_init_est = vx_false_e;
        vx_scalar s_lk_use_init_est = vxCreateScalar(context_, VX_TYPE_BOOL, &lk_use_init_est);
        NVXIO_CHECK_REFERENCE(s_lk_use_init_est);

        // RGB to Y conversion nodes
        //cvt_color_node_ = vxColorConvertNode(main_graph_, frame, frameGray);
        //NVXIO_CHECK_REFERENCE(cvt_color_node_);

        // Pyramid image node
        pyr_node_ = vxGaussianPyramidNode(main_graph_, frameGray, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0));
        NVXIO_CHECK_REFERENCE(pyr_node_);
        //vxOpticalFlowPyrLKNode
        opt_flow_node_ = vxOpticalFlowPyrLKNode(main_graph_,
            (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1), (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),
            opt_in, opt_in,
            opt_feature, VX_TERM_CRITERIA_BOTH, s_lk_epsilon, s_lk_num_iters, s_lk_use_init_est, params_.lk_win_size);
        NVXIO_CHECK_REFERENCE(opt_flow_node_);

        // Extended Harris corner node
        feature_track_node_ = nvxHarrisTrackNode(main_graph_, frameGray, harris_feature, mask,
            opt_feature, params_.harris_k, params_.harris_thresh, params_.harris_cell_size, NULL);
        NVXIO_CHECK_REFERENCE(feature_track_node_);
        std::cout<<"params_.harris_k   "<<params_.harris_k<<std::endl;
/*
        //vxOpticalFlowPyrLKNode
        opt_flow_node_ = vxOpticalFlowPyrLKNode(main_graph_,
            (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1), (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),
            (vx_array)vxGetReferenceFromDelay(pts_delay_, -1), (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
            kp_curr_list_, VX_TERM_CRITERIA_BOTH, s_lk_epsilon, s_lk_num_iters, s_lk_use_init_est, params_.lk_win_size);
        NVXIO_CHECK_REFERENCE(opt_flow_node_);

        // Extended Harris corner node
        feature_track_node_ = nvxHarrisTrackNode(main_graph_, frameGray, (vx_array)vxGetReferenceFromDelay(pts_delay_, 0), mask,
            kp_curr_list_, params_.harris_k, params_.harris_thresh, params_.harris_cell_size, NULL);
        NVXIO_CHECK_REFERENCE(feature_track_node_);
        std::cout<<"params_.harris_k   "<<params_.harris_k<<std::endl;
*/

        // Graph verification.
        // Note: This verification is mandatory prior to graph execution.
        NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_) );

        vxReleaseScalar(&s_lk_epsilon);
        vxReleaseScalar(&s_lk_num_iters);
        vxReleaseScalar(&s_lk_use_init_est);
        vxReleaseImage(&frameGray);
    }
}

nvx::FeatureTracker::HarrisPyrLKParams::HarrisPyrLKParams()
{
    // parameters for optical flow node
    pyr_levels = 6;      //6
    lk_num_iters = 30;    //5
    lk_win_size = 5;    //10

    // parameters for harris_track node
    harris_k = 0.08f;             //0.04f
    harris_thresh = 100.0f;       //100.0f
    harris_cell_size = 30;        //18
    array_capacity = 2000;        //2000
}

nvx::FeatureTracker* nvx::FeatureTracker::createHarrisPyrLK(vx_context context, const HarrisPyrLKParams& params)
{

    std::cout<<"here params.harris_k          "<<params.harris_k<<std::endl;
    std::cout<<"here params.harris_thresh     "<<params.harris_thresh<<std::endl;
    std::cout<<"here params.harris_cell_size  "<<params.harris_cell_size<<std::endl;
    std::cout<<"here params.array_capacity    "<<params.array_capacity<<std::endl<<std::endl;
    std::cout<<"here params.pyr_levels        "<<params.pyr_levels<<std::endl;
    std::cout<<"here params.lk_num_iters      "<<params.lk_num_iters<<std::endl;
    std::cout<<"here params.lk_win_size       "<<params.lk_win_size<<std::endl;


    return new FeatureTrackerHarrisPyrLK(context, params);
}
