#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

//For debug only
#include <opencv2/highgui.hpp>
#include <iostream>

#include <vector>

#include "laptopVR.hpp"

void LaptopVR::setFlipFrame(bool flip)
{
    flipFrame = flip;
}

void LaptopVR::setDpi(float d)
{
    if (d > 0)
    {
        dpi = d;
    }
    else
    {
        dpi = 96.0;
    }
}

cv::Point LaptopVR::getRectCenter(cv::Rect rect)
{
    return (rect.tl() + rect.br()) / 2;
}

cv::Scalar LaptopVR::projectPointOnFrame(cv::Scalar point, cv::Scalar observer)
{
    //parameter indicating position on the point-observer line
    float alpha;
    cv::Scalar proj;
    
    //search for the point on point-observer line with coordinate Z=0
    alpha = observer[2]/(observer[2]-point[2]);
    
    //calculate all other coordinates
    proj[0] = alpha*point[0] + (1.f-alpha)*observer[0];
    proj[1] = alpha*point[1] + (1.f-alpha)*observer[1];
    proj[2] = 0.f;
    return proj;
}

Line LaptopVR::projectLineOnFrame(Line line, cv::Scalar observer)
{
    Line proj;
    
    //project start and end points
    proj.p1 = projectPointOnFrame(line.p1, observer);
    proj.p2 = projectPointOnFrame(line.p2, observer);
    return proj;
}

LaptopVR::LaptopVR(int width, int height)
{
    //init box width & height if parameters are ok
    //else set default
    if (width>0)
    {
        boxWidth = static_cast<float>(width);
    }
    else
    {
        boxWidth = 640.f;
    }
    
    if (height>0)
    {
        boxHeight = static_cast<float>(height);
    }
    else
    {
        boxHeight = 320.f;
    }
    
    //default box depth
    boxDepth = boxWidth;
    
    //default number of lines
    nLinesRight = 11;
    nLinesTop = 15;
    
    //default size of webcam frames
    camWidth = 640;
    camHeight = 480;
    
    //default dpi
    dpi = 96.0;
    
    //calculate approx real face width in pixels from its width in cm
    realFaceWidth = 18.5 * CM_TO_INCH * dpi; //18.5
    
    //set default camera fov parameters
    z2xCamRatio = 17.0 / 16.0;
    z2yCamRatio = z2xCamRatio * camWidth /camHeight;
    
    //Create and init render frame
    renderCanvas.create((int)boxHeight, (int)boxWidth, CV_8UC3);
    renderCanvas = 0;
    
    //by default frame is not flipped
    flipFrame = false;
        
}

bool LaptopVR::init()
{
    //load face detector
    if (!faceDetectorHaar.load("./resource/haarcascade_frontalface_default.xml"))
    {
        printf("Cannot load Haar face detector!\n");
        return false;
    }
    
    generateLines();
    
    observerKalmanFilter.init(6,3,0);
    
    observerKalmanFilter.transitionMatrix = 0;
    observerKalmanFilter.transitionMatrix.diag(0) = 1.;
    observerKalmanFilter.transitionMatrix.diag(3) = 1.;
    
    cv::setIdentity(observerKalmanFilter.measurementMatrix);
    
    observerKalmanFilter.processNoiseCov = 0;
    observerKalmanFilter.processNoiseCov.at<float>(0,0) = 0.001;
    observerKalmanFilter.processNoiseCov.at<float>(1,1) = 0.001;
    observerKalmanFilter.processNoiseCov.at<float>(2,2) = 0.001;
    observerKalmanFilter.processNoiseCov.at<float>(3,3) = 0.001;
    observerKalmanFilter.processNoiseCov.at<float>(4,4) = 0.001;
    observerKalmanFilter.processNoiseCov.at<float>(5,5) = 0.001;
    
    std::cout<<observerKalmanFilter.processNoiseCov<<std::endl;
    
    return true;
}

void LaptopVR::generateLines()
{
    int i;
    Line l;
    //lines on right & left, horisontal lines on back
    for(i=0; i<nLinesRight; i++)
    {
        l.p1 = cv::Scalar(0, boxHeight*i/(nLinesRight-1), 0);
        l.p2 = cv::Scalar(0, boxHeight*i/(nLinesRight-1), boxDepth);
        boxLines.push_back(l);
        l.p1 = cv::Scalar(boxWidth, boxHeight*i/(nLinesRight-1), 0);
        l.p2 = cv::Scalar(boxWidth, boxHeight*i/(nLinesRight-1), boxDepth);
        boxLines.push_back(l);
        l.p1 = cv::Scalar(0, boxHeight*i/(nLinesRight-1), boxDepth);
        l.p2 = cv::Scalar(boxWidth, boxHeight*i/(nLinesRight-1), boxDepth);
        boxLines.push_back(l);
    }
    //lines on top & bottom, vertical lines on back
    for(i=0; i<nLinesTop; i++)
    {
        l.p1 = cv::Scalar(boxWidth*i/(nLinesTop-1), 0, 0);
        l.p2 = cv::Scalar(boxWidth*i/(nLinesTop-1), 0, boxDepth);
        boxLines.push_back(l);
        l.p1 = cv::Scalar(boxWidth*i/(nLinesTop-1), boxHeight, 0);
        l.p2 = cv::Scalar(boxWidth*i/(nLinesTop-1), boxHeight, boxDepth);
        boxLines.push_back(l);
        l.p1 = cv::Scalar(boxWidth*i/(nLinesTop-1), 0, boxDepth);
        l.p2 = cv::Scalar(boxWidth*i/(nLinesTop-1), boxHeight, boxDepth);
        boxLines.push_back(l);
    }
    return;
    
}

cv::Mat LaptopVR::renderNextFrame(cv::Scalar position)
{
    //fill frame with black
    renderCanvas = 0;
    
    Line line;
    
    //Project and draw all lines
    for (size_t i=0; i<boxLines.size(); i++)
    {
        line = projectLineOnFrame(boxLines[i], position);
        cv::line(renderCanvas, cv::Point(line.p1[0], line.p1[1]), cv::Point(line.p2[0], line.p2[1]), cv::Scalar(0,255,0), 2);
    }
    
    return renderCanvas;
}

cv::Scalar LaptopVR::getObserverFromFace(cv::Rect face, cv::Point eye)
{
    float depth;
    cv::Point coord;
    
    //calculate frame depth in px from width of detected face
    depth = realFaceWidth * camWidth * z2xCamRatio / face.width;
    
    //move (0,0) point to the center of the frame
    coord = eye - cv::Point(camWidth, camHeight) / 2;
    
    //conver from camera space to monitor space
    coord *= (realFaceWidth/face.width);
    
    //mirror observer position if we need to flip camera frame
    if (flipFrame)
    {
        coord.x = -coord.x;
    }
    
    //add shift of the webcam (it is in the top center point of the screen)
    coord.x += boxWidth/2;
    
    return cv::Scalar(coord.x, coord.y, -depth);
    
}

cv::Scalar LaptopVR::detectObserver(cv::Mat scene)
{
    cv::Mat1b sceneGray;
    std::vector<cv::Rect> detections;
    cv::Rect bestDet;
    cv::Scalar observer;
    cv::Point camObserver;
    
    //refresh camera frames size
    camWidth = scene.cols;
    camHeight = scene.rows;
    
    //conver to gray for detector
    cvtColor(scene, sceneGray, CV_BGR2GRAY);
    
    //detect facec
    faceDetectorHaar.detectMultiScale(sceneGray, detections);
    
    //if we have faces
    if (detections.size() > 0)
    {
        int width = 0;
        int ind = 0;
        
        //pick biggest face
        for (size_t i=0; i<detections.size(); i++)
        {
            if(detections[i].width > width)
            {
                width = detections[i].width;
                ind = i;
            }
        }
        bestDet = detections[ind];
        
        //calculate eye position
        camObserver = bestDet.tl() + cv::Point(bestDet.width/2, bestDet.height/3);
        
        //get observer coordinates
        observer = getObserverFromFace(bestDet, camObserver);
        
        //cv::rectangle(scene, bestDet, cv::Scalar(255,0,0));
        //cv::circle(scene, camObserver, 3, cv::Scalar(0,255,0), -1);
        
        return observer;
    }
    
    //cv::imshow("gray", scene);
    
    return cv::Scalar();
}

cv::Mat LaptopVR::renderFromCamFrame(cv::Mat camFrame)
{
    cv::Scalar observer;
    
    observer = detectObserver(camFrame);
    if (observer != cv::Scalar())
    {
        sceneObserver = observer;
        return renderNextFrame(sceneObserver);
    }
    return renderCanvas;
    
}