#include "feature_tracker.h"

FeatureTracker::FeatureTracker()
{
    cnt = 0;
    mask = NULL;
    isInit = false;
    id_count = 0;
}

void FeatureTracker::changeType(vx_array vx_pt, vector<cv::Point2f> &cv_pt )
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
}

void FeatureTracker::addFeatures()
{
    double minDistance = MIN_DIST;

    const int cell_size = minDistance;
    const int grid_width = (COL + cell_size - 1) / cell_size;
    const int grid_height = (ROW + cell_size - 1) / cell_size;
    minDistance *= minDistance;

    vector<vector<cv::Point2f>> grid(grid_width * grid_height);
    vector<cv::Point2f> new_cur_pts;
    vector<int> new_cur_track_cnt;
    vector<int> new_cur_ids;
    int prev_num, now_num;
    prev_num = cur_pts.size();

    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        cv::Point2f p = cv::Point2f(cur_pts[i].x, cur_pts[i].y);
        bool good = true;
        int x_cell = p.x / cell_size;
        int y_cell = p.y / cell_size;
        int x1 = x_cell - 1;
        int y1 = y_cell - 1;
        int x2 = x_cell + 1;
        int y2 = y_cell + 1;

        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(grid_width - 1 , x2);
        y2 = std::min(grid_height - 1, y2);

        for (int yy = y1; yy <= y2; yy++)
            for (int xx = x1; xx <= x2; xx++)
                for (auto & pp : grid[yy * grid_width + xx])
                {
                    float dx = p.x - pp.x;
                    float dy = p.y - pp.y;
                    if (dx * dx + dy * dy < minDistance)
                    {
                        good = false;
                        goto break_out1;
                    }
                }
break_out1:
        if (good)
        {
            grid[y_cell * grid_width + x_cell].push_back(p);
            new_cur_pts.push_back(p);
            new_cur_ids.push_back(cur_ids[i]);
            new_cur_track_cnt.push_back(cur_track_cnt[i]);
        }
    }
    now_num = new_cur_pts.size();
    //ROS_INFO("Delete %d features", prev_num - now_num);

    prev_num = now_num;
    for (auto & p : harris_pts)
    {
        if ((int)new_cur_pts.size() >= 150)
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
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(grid_width - 1 , x2);
        y2 = std::min(grid_height - 1, y2);

        for (int yy = y1; yy <= y2; yy++)
            for (int xx = x1; xx <= x2; xx++)
                for (auto & pp : grid[yy * grid_width + xx])
                {
                    float dx = p.x - pp.x;
                    float dy = p.y - pp.y;
                    if (dx * dx + dy * dy < minDistance)
                    {
                        good = false;
                        goto break_out2;
                    }
                }
break_out2:
        if (good)
        {
            grid[y_cell * grid_width + x_cell].push_back(p);
            new_cur_pts.push_back(p);
            new_cur_ids.push_back(id_count);
            id_count++;
            if (id_count >= MAX_CNT)
                id_count = 0;
            new_cur_track_cnt.push_back(1);
        }
    }
    now_num = new_cur_pts.size();
    //ROS_INFO("add %d features", now_num - prev_num);
    cur_pts = new_cur_pts;
    cur_track_cnt = new_cur_track_cnt;
    cur_ids = new_cur_ids;
}

void FeatureTracker::printresult()
{
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cout << "id  " << cur_ids[i] << "  track_num  " << cur_track_cnt[i] << " points x " << cur_pts[i].x << " y " << cur_pts[i].y << endl;
}

void FeatureTracker::printvector(vector<cv::Point2f> &v)
{
    for (unsigned int i = 0; i < v.size(); i++)
        cout << "id  " << i << "  x " << v[i].x << "  y " << v[i].y << endl;
    cout << "print finish" << endl;
}

vector<cv::Point2f> FeatureTracker::undistortedPoints(std::vector<cv::Point2f> v)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < v.size(); i++)
    {
        Eigen::Vector2d a(v[i].x, v[i].y), re_pro;
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        if (std::isnan(b.norm()))
            ROS_WARN("nan false");
    }
    return un_pts;
}

vector<cv::Point3f> FeatureTracker::undistortedPoints_pub(std::vector<cv::Point2f> v)
{
    vector<cv::Point3f> un_pts;
    for (unsigned int i = 0; i < v.size(); i++)
    {
        Eigen::Vector2d a(v[i].x, v[i].y), a_c(v[i].x + 0.01, v[i].y),re_pro;
        Eigen::Vector3d b, c;
        Eigen::Vector2d b_n, c_n;
        m_camera->liftProjective(a, b);
        if (std::isnan(b.norm()) || b.z() < 1e-3)
            un_pts.push_back(cv::Point3f(0.0, 0.0, 0.0));
        else
        {
            m_camera->liftProjective(a_c, c);
            b_n = Eigen::Vector2d(b.x()/b.z(), b.y()/b.z());
            c_n = Eigen::Vector2d(c.x()/c.z(), c.y()/c.z());
            un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 100 * abs(b_n.norm() - c_n.norm())));
        }
    }
    return un_pts;
}

void FeatureTracker::ransac(std::vector<cv::Point2f> prev, std::vector<cv::Point2f> curr)
{
    //ROS_INFO("ransac begin!");
    if ((int)curr.size() < 4 )
        return;
    vector<uchar> status;
    // ransac after undistort
    std::vector<cv::Point2f> prev_un = undistortedPoints(prev);
    std::vector<cv::Point2f> curr_un = undistortedPoints(curr);
    //cv::findFundamentalMat(prev_un, curr_un, cv::FM_RANSAC, ransac_thres, 0.99, status);
    static double rans_t = 1.0 / 200.0; //--> on feature
    cv::findFundamentalMat(prev_un, curr_un, cv::FM_RANSAC, rans_t, 0.99, status);
    int num = curr.size();
    cur_pts.clear();
    cur_ids.clear();
    cur_track_cnt.clear();

    for (unsigned int i = 0; i < curr.size(); i++)
    {
        if ((int)status[i] == 1)
        {
            if (curr[i].y < 5 || curr[i].y > ROW - 5 || curr[i].x < 5 || curr[i].x > COL - 5 ||
                    (curr[i].x - COL / 2.0) * (curr[i].x - COL / 2.0)
                    + (curr[i].y - ROW / 2.0) * (curr[i].y - ROW / 2.0) > 345 * 345 )
                continue;
            cur_pts.push_back(curr[i]);
            cur_ids.push_back(ransac_ids[i]);
            cur_track_cnt.push_back(ransac_track_cnt[i]);
        }
    }
    ROS_INFO("RANSAC delete %d features" , num - (int)cur_pts.size());
}




