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

cv::Rect LaptopVR::scaleRectCentered(cv::Rect rect, float factor)
{
    cv::Rect res;
    res.x = rect.x - rect.width*(factor-1)*0.5;
    res.y = rect.y - rect.height*(factor-1)*0.5;
    res.width = rect.width*factor;
    res.height = rect.height*factor;
    return res;
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

LaptopVR::LaptopVR(int width, int height):
landmarkDetector("./resource/seeta_fa_v1.1.bin")
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
    
    isFirstFrame = true;
    
    detectionROI = cv::Rect(0,0,camWidth,camHeight);
    minFaceWidth = camHeight/3;
    maxFaceWidth = camHeight/3;
    expandFactorROI = 1.5;
        
}

bool LaptopVR::init()
{
    //load face detector
    if (!faceDetectorHaar.load("./resource/lbpcascade_frontalface.xml"))
    {
        printf("Cannot load Haar face detector!\n");
        return false;
    }
    
    generateLines();
    
    //6 state coordinates: x,y,z,vx,vy,vz
    //3 measurment coordinates: x,y,z
    observerKalmanFilter.init(6,3);
    
    //Physical model of the process
    //Standard kinetic model: movement with constant speed
    //state(k+1) = transitionMatrix * state(k)
    observerKalmanFilter.transitionMatrix = 0;
    observerKalmanFilter.transitionMatrix.diag(0) = 1.;
    observerKalmanFilter.transitionMatrix.diag(3) = 1.;
    
    //we measure only x,y,z
    //measurment(k) = measurementMatrix * state(k)
    cv::setIdentity(observerKalmanFilter.measurementMatrix);
    
    observerKalmanFilter.processNoiseCov = 0;
    observerKalmanFilter.processNoiseCov.at<float>(0,0) = 1.;
    observerKalmanFilter.processNoiseCov.at<float>(1,1) = 1.;
    observerKalmanFilter.processNoiseCov.at<float>(2,2) = 1.;
    observerKalmanFilter.processNoiseCov.at<float>(3,3) = 10.;
    observerKalmanFilter.processNoiseCov.at<float>(4,4) = 10.;
    observerKalmanFilter.processNoiseCov.at<float>(5,5) = 10.;
    
    observerKalmanFilter.measurementNoiseCov = 0;
    observerKalmanFilter.measurementNoiseCov.at<float>(0,0) = 150.;
    observerKalmanFilter.measurementNoiseCov.at<float>(1,1) = 150.;
    observerKalmanFilter.measurementNoiseCov.at<float>(2,2) = 2000.;
    
    observerKalmanFilter.errorCovPost = 0;
    observerKalmanFilter.errorCovPost.at<float>(0,0) = 450.;
    observerKalmanFilter.errorCovPost.at<float>(1,1) = 450.;
    observerKalmanFilter.errorCovPost.at<float>(2,2) = 4000.;
    observerKalmanFilter.errorCovPost.at<float>(3,3) = 4000.;
    observerKalmanFilter.errorCovPost.at<float>(4,4) = 4000.;
    observerKalmanFilter.errorCovPost.at<float>(5,5) = 4000.;
    
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


cv::Scalar LaptopVR::detectObserverSeeta(cv::Mat scene)
{
    cv::Mat1b sceneGray;
    std::vector<cv::Rect> detections;
    cv::Rect bestDet;
    cv::Scalar observer;
    cv::Point camObserver;
    
    //refresh camera frames size
    camWidth = scene.cols;
    camHeight = scene.rows;
    
    detectionROI = scaleRectCentered(detectionROI, expandFactorROI) & cv::Rect(0, 0, camWidth, camHeight);
    maxFaceWidth = std::max(maxFaceWidth*expandFactorROI, (float)camHeight);
    minFaceWidth = std::min(minFaceWidth/expandFactorROI, (float)15.);
    
    //conver to gray for detector
    cvtColor(scene, sceneGray, CV_BGR2GRAY);
    
    //detect facec
    faceDetectorHaar.detectMultiScale(sceneGray(detectionROI), detections, 1.15, 3, 0, 
                                      cv::Size(minFaceWidth, minFaceWidth), cv::Size(maxFaceWidth, maxFaceWidth));
    
    //cv::rectangle(scene, detectionROI, cv::Scalar(0,0,255), 1);
    
    //if we have faces
    if (detections.size() > 0)
    {
        int width = 0;
        int ind = 0;
        
        //pick biggest face
        for (size_t i=0; i<detections.size(); i++)
        {            
            //cv::rectangle(scene, detections[i], cv::Scalar(255,0,0));
            
            if(detections[i].width > width)
            {
                width = detections[i].width;
                ind = i;
            }
        }
        bestDet = detections[ind] + detectionROI.tl();
        
        detectionROI = bestDet;
        minFaceWidth = bestDet.width;
        maxFaceWidth = bestDet.width;
        
        seeta::FaceInfo SeetaDet;
        SeetaDet.bbox.x = bestDet.x;
        SeetaDet.bbox.y = bestDet.y;
        SeetaDet.bbox.width = bestDet.width;
        SeetaDet.bbox.height = bestDet.height;
        
        seeta::ImageData imageData;
        imageData.data = sceneGray.data;
        imageData.height = sceneGray.rows;
        imageData.width = sceneGray.cols;
        imageData.num_channels = 1;
        
        seeta::FacialLandmark landmarks[5];
        landmarkDetector.PointDetectLandmarks(imageData, SeetaDet, landmarks);
        
        camObserver = cv::Point((landmarks[0].x+landmarks[1].x)/2, (landmarks[0].y+landmarks[1].y)/2);
        
        //get observer coordinates
        observer = getObserverFromFace(bestDet, camObserver);
        
        /*
        cv::rectangle(scene, bestDet, cv::Scalar(255,0,0));
        for(int i=0; i<2; i++)
        {
            cv::circle(scene, cv::Point(landmarks[i].x, landmarks[i].y), 3, cv::Scalar(0,0,255), -1);
        }*/
        //cv::circle(scene, camObserver, 3, cv::Scalar(0,255,0), -1);
        
        return observer;
    }
    
    //cv::imshow("gray", scene);
    
    return cv::Scalar();
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
    
    detectionROI = scaleRectCentered(detectionROI, expandFactorROI) & cv::Rect(0, 0, camWidth, camHeight);
    maxFaceWidth = std::max(maxFaceWidth*expandFactorROI, (float)camHeight);
    minFaceWidth = std::min(minFaceWidth/expandFactorROI, (float)15.);
    
    //conver to gray for detector
    cvtColor(scene, sceneGray, CV_BGR2GRAY);
    
    //detect facec
    faceDetectorHaar.detectMultiScale(sceneGray(detectionROI), detections, 1.15, 3, 0, 
                                      cv::Size(minFaceWidth, minFaceWidth), cv::Size(maxFaceWidth, maxFaceWidth));
    
    //cv::rectangle(scene, detectionROI, cv::Scalar(0,0,255), 1);
    
    //if we have faces
    if (detections.size() > 0)
    {
        int width = 0;
        int ind = 0;
        
        //pick biggest face
        for (size_t i=0; i<detections.size(); i++)
        {            
            //cv::rectangle(scene, detections[i], cv::Scalar(255,0,0));
            
            if(detections[i].width > width)
            {
                width = detections[i].width;
                ind = i;
            }
        }
        bestDet = detections[ind] + detectionROI.tl();
        
        detectionROI = bestDet;
        minFaceWidth = bestDet.width;
        maxFaceWidth = bestDet.width;
        
        //calculate eye position
        camObserver = bestDet.tl() + cv::Point(bestDet.width/2, bestDet.height/3);
        
        //get observer coordinates
        observer = getObserverFromFace(bestDet, camObserver);
        
        cv::rectangle(scene, bestDet, cv::Scalar(255,0,0));
        //cv::circle(scene, camObserver, 3, cv::Scalar(0,255,0), -1);
        
        return observer;
    }
    
    //cv::imshow("gray", scene);
    
    return cv::Scalar();
}

cv::Scalar LaptopVR::filterObserverPosition(cv::Scalar observer)
{
    cv::Mat_<float> measurement(3,1);
    cv::Scalar filtered;
    
    if(isFirstFrame)
    {
        isFirstFrame = false;
        observerKalmanFilter.statePre = 0;
        
        if(observer == cv::Scalar())
        {
            observerKalmanFilter.statePre.at<float>(0) = boxWidth/2;
            observerKalmanFilter.statePre.at<float>(1) = boxHeight/2;
            observerKalmanFilter.statePre.at<float>(2) = boxWidth;
            
            observerKalmanFilter.errorCovPost *= 100;
        }
        else
        {
            observerKalmanFilter.statePre.at<float>(0) = observer[0];
            observerKalmanFilter.statePre.at<float>(1) = observer[1];
            observerKalmanFilter.statePre.at<float>(2) = observer[2];
        }
    }
    
    cv::Mat prediction = observerKalmanFilter.predict();
    
    if(observer != cv::Scalar())
    {
        cv::Mat refined_prediction;
        
        measurement.at<float>(0) = observer[0];
        measurement.at<float>(1) = observer[1];
        measurement.at<float>(2) = observer[2];
        
        refined_prediction = observerKalmanFilter.correct(measurement);
        
        filtered[0] = refined_prediction.at<float>(0);
        filtered[1] = refined_prediction.at<float>(1);
        filtered[2] = refined_prediction.at<float>(2);
    }
    else
    {
        filtered[0] = prediction.at<float>(0);
        filtered[1] = prediction.at<float>(1);
        filtered[2] = prediction.at<float>(2);
    }
    
#if KALMAN_DEBUG
    xraw.push_back(observer[0]);
    yraw.push_back(observer[1]);
    zraw.push_back(observer[2]);
    
    xfil.push_back(filtered[0]);
    yfil.push_back(filtered[1]);
    zfil.push_back(filtered[2]);
#endif
    
    return filtered;
}

cv::Mat LaptopVR::renderFromCamFrame(cv::Mat camFrame)
{
    cv::Scalar observer;
    
    observer = detectObserverSeeta(camFrame);
    observer = filterObserverPosition(observer);
    if (observer != cv::Scalar())
    {
        sceneObserver = observer;
        return renderNextFrame(sceneObserver);
    }
    return renderCanvas;
    
}
