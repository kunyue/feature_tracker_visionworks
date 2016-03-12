cmake_minimum_required(VERSION 2.8.3)
project(visionworks)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv;/usr/include")

find_package(catkin REQUIRED COMPONENTS
    roscpp
	cv_bridge
    std_msgs
	camera_model
	)
find_package(OpenCV REQUIRED)
find_package(VisionWorks REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${VisionWorks_INCLUDE_DIRS} /usr/include/nvxio/include /usr/include/3rdparty/opengl  /usr/include/libdrm /usr/include/freetype2   /usr/local/cuda-6.5/include   /usr/include/opencv   /usr/local/cuda-6.5/include   /usr/include/3rdparty/eigen)

catkin_package(
    )

add_executable(feature_tracker 
	src/feature_tracker_node.cpp 
	src/nvx_feature_tracker.cpp
	src/feature_tracker.cpp)



target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS} ${VisionWorks} libnvxio.a -lfreetype   -lGL -lglfw   -lXrandr -lXi -lXxf86vm -lX11   -lgstpbutils-1.0 -lgstapp-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0    -L/usr/local/cuda-6.5/lib -lvisionworks -lcudart   -L/usr/local/cuda-6.5/lib -L/usr/local/cuda-6.5/lib/stubs -lcudart -lcublas -lcufft -lnppc -lnppi -lnpps   -L/usr/local/cuda-6.5/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts  -lcufft -lnpps -lnppi -lnppc -lcudart -ltbb -lrt -lpthread -lm -ldl)
#target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS} ${VisionWorks} libnvxio.a -lfreetype   -lGL -lglfw   -lXrandr -lXi -lXxf86vm -lX11   -lgstpbutils-1.0 -lgstapp-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0    -L/usr/local/cuda-6.5/lib -lvisionworks -lcudart   -L/usr/local/cuda-6.5/lib -L/usr/local/cuda-6.5/lib/stubs -lcudart -lcublas -lcufft -lnppc -lnppi -lnpps   -L/usr/local/cuda-6.5/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_detection_based_tracker -lopencv_esm_panorama -lopencv_facedetect -lopencv_imuvstab -lopencv_tegra -lopencv_vstab -lcufft -lnpps -lnppi -lnppc -lcudart -ltbb -lrt -lpthread -lm -ldl)

