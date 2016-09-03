#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>

#define KALMAN_DEBUG 1

#define CM_TO_INCH 0.393701

/**
 * 3D Line class - for convenience
 */
class Line
{
public:
    //start/end points
    cv::Scalar p1, p2;
};

/*
 * LaptopVR engine
 * class including methods to detect observer from webcam and render VR environment
 */
class LaptopVR
{
public:
    /**
     * Constructor
     * @param width : width of the screen in pixels
     * @param height : height of the screen in pixels
     */
    LaptopVR(int width=640, int height=480);
    
    /**
     * Init LaptopVR engine
     * @return true, if successfull, false otherwise
     */
    bool init();
    
    /**
     * Render a frame for given observer
     * @param position : position of the observer, relative to the screen, in pixels
     * @return rendered frame
     * @note (0,0,0) point is in the top left corner of the screen, positive direction of Z axis points "inside" the monitor
     */
    cv::Mat renderNextFrame(cv::Scalar position);
    
    /**
     * Detect observer on given frame
     * @param scene : frame from webcam to detect observer
     * @return position of the observer
     */
    cv::Scalar detectObserver(cv::Mat scene);
    
    /** 
     * Perspective projection of a point onto the monitor plane, relative for observer
     * @param point : point to project
     * @param observer : observer coordinates (center for projection)
     * @return projection
     */
    cv::Scalar projectPointOnFrame(cv::Scalar point, cv::Scalar observer);
    
    /** 
     * Perspective projection of a line onto the monitor plane, relative for observer
     * @param line : line to project
     * @param observer : observer coordinates (center for projection)
     * @return projection
     */
    Line projectLineOnFrame(Line line, cv::Scalar observer);
    
    /**
     * Get center of a rectangle
     * @param rect : rectangle
     * @return the center
     */
    cv::Point getRectCenter(cv::Rect rect);
    
    /**
     * Scale rect around center
     * @param rect rectangle
     * @param factor scale factor
     * @return scaled rect
     */
    cv::Rect scaleRectCentered(cv::Rect rect, float factor);
    
    /**
     * Set flag flipFrame
     * @param flip : true if necessary to flip frame, false otherwise
     */
    void setFlipFrame(bool flip);
    
    /**
     * render VR frame from webcam frame
     * @param camFrame frome obtained from webcam
     * @return rendered vr frame
     */
    cv::Mat renderFromCamFrame(cv::Mat camFrame);
    
    /**
     * set dpi of the monitor
     * @param d new dpi
     */
    void setDpi(float d);
  
        
#if KALMAN_DEBUG
    std::vector<double> xraw, yraw, zraw, xfil, yfil, zfil; 
#endif
    
private:
    
    cv::Rect detectionROI;
    int minFaceWidth;
    int maxFaceWidth;
    float expandFactorROI;
   
    bool isFirstFrame;
    
    cv::KalmanFilter observerKalmanFilter;
    
    /**
     * perform Kalman filtering on observer position
     * @param observer : observer position
     * @return filtered position
     */
    cv::Scalar filterObserverPosition(cv::Scalar observer);
    
    //observer of the scene
    cv::Scalar sceneObserver;
    
    //if to flip camera frame arount vertical axis
    //frame itself is not flipped: observer position is mirrored
    bool flipFrame;
    
    //box elements
    //measured in pixels
    float boxDepth, boxWidth, boxHeight;
    
    //dpi
    float dpi;
    
    //approximate width of a real face in pixels (computed from real face width in cm)
    float realFaceWidth;
    
    //[frame depth]/[frame width] and [frame depth]/[frame height] for camera
    //calculated from camera angles of view
    float z2xCamRatio, z2yCamRatio;
    
    //width and height of frames obtained from camera, in pixels
    int camWidth, camHeight;
    
    //frame to render
    cv::Mat renderCanvas;

    //lines to render on frame
    std::vector<Line> boxLines;
    //number of lines on top and right
    int nLinesTop, nLinesRight;
    
    /**
     * generate lines to draw
     */
    void generateLines();
    
    //frontal Haar-based face detector
    cv::CascadeClassifier faceDetectorHaar;
    
    /**
     * Calculate observer coordinates from a face and position of eyes
     * @param face : rect which describes a detected face
     * @param eye : a point inside face rect considered as a point between two eyes (observer point)
     * its coordinates a relative to camera frame, not face rect
     * @return observer coordinates
     */
    cv::Scalar getObserverFromFace(cv::Rect face, cv::Point eye);
    
};
