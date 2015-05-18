cmake_minimum_required (VERSION 2.6 FATAL_ERROR)

project      (gesture_recognition)
find_package (Qt4 REQUIRED)
find_package (PCL 1.7.1 REQUIRED)
find_package (OpenCV REQUIRED)

include_directories (${PCL_INCLUDE_DIRS})
link_directories    (${PCL_LIBRARY_DIRS})
add_definitions     (${PCL_DEFINITIONS})
add_definitions     (${OPENCV_DEFINITIONS})

set  (SOURCES 
	main.cpp
	FaceDetection.cpp
	HandGesture.cpp
        #image_viewer_threaded.cpp
	SkinDetection.cpp
	StateObserver.cpp
	Viewer.cpp
)

set  (HEADERS 
	common.h
	FaceDetection.h
	HandGesture.h
        #image_viewer_threaded.h
	SkinDetection.h
	StateObserver.h
	Viewer.h
)

set  (FORMS   
)

QT4_WRAP_CPP (HEADERS_MOC   ${HEADERS})
QT4_WRAP_UI  (FORMS_HEADERS ${FORMS})

INCLUDE         (${QT_USE_FILE})
ADD_DEFINITIONS (${QT_DEFINITIONS})

ADD_EXECUTABLE  (gesture_recognition ${SOURCES}
                                ${FORMS_HEADERS}
                                ${HEADERS_MOC})

TARGET_LINK_LIBRARIES (gesture_recognition ${QT_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${OpenCV_LIBS})

