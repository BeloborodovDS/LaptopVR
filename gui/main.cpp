#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <GLFW/glfw3.h>

#include <iostream>

#include <ctime>

#include "../src/laptopVR.hpp"

//debug
#include <vector>

using namespace std;
using namespace cv;

Mat plotVectors(vector<double> x1, vector<double> y1, 
                vector<double> x2 = vector<double>(), vector<double> y2 = vector<double>())
{
    Mat plot(480, 640, CV_8UC3);
    plot = 0;
    
    double xmn1=1000000, xmx1=-1000000;
    double ymn1=1000000, ymx1=-1000000;
    double xmn2=1000000, xmx2=-1000000;
    double ymn2=1000000, ymx2=-1000000;
    for(int i=0; i<x1.size(); i++)
    {
        if(x1[i]<xmn1)
            xmn1=x1[i];
        if(x1[i]>xmx1)
            xmx1=x1[i];
        
        if(y1[i]<ymn1)
            ymn1=y1[i];
        if(y1[i]>ymx1)
            ymx1=y1[i];
    }
    for(int i=0; i<x2.size(); i++)
    {
        if(x2[i]<xmn2)
            xmn2=x2[i];
        if(x2[i]>xmx2)
            xmx2=x2[i];
        
        if(y2[i]<ymn2)
            ymn2=y2[i];
        if(y2[i]>ymx2)
            ymx2=y2[i];
    }
    double xmx = max(xmx1, xmx2);
    double ymx = max(ymx1, ymx2);
    double xmn = min(xmn1, xmn2);
    double ymn = min(ymn1, ymn2);
    
    Point pnt1, pnt2;
    
    for(int i=0; i<x1.size()-1; i++)
    {
        pnt1 = Point((x1[i]-xmn)/(xmx-xmn)*640, (y1[i]-ymn)/(ymx-ymn)*480);
        pnt2 = Point((x1[i+1]-xmn)/(xmx-xmn)*640, (y1[i+1]-ymn)/(ymx-ymn)*480);
        line(plot, pnt1, pnt2, Scalar(0,255,0), 2);
    }
    for(int i=0; i<x2.size()-1; i++)
    {
        pnt1 = Point((x2[i]-xmn)/(xmx-xmn)*640, (y2[i]-ymn)/(ymx-ymn)*480);
        pnt2 = Point((x2[i+1]-xmn)/(xmx-xmn)*640, (y2[i+1]-ymn)/(ymx-ymn)*480);
        line(plot, pnt1, pnt2, Scalar(0,0,255), 2);
    }
    
    return plot;
}

int main()
{
    
    clock_t start;
    int nframes=0;
    
    //Init GLFW
    
    if(!glfwInit())
    {
        cout<<"GLFW init failed!"<<endl;
        return 0;
    }
    
    //Get primary minitor
    
    GLFWmonitor* monitor;
    if((monitor=glfwGetPrimaryMonitor()) == NULL)
    {
        cout<<"GLFW failed to find primary monitor!"<<endl;
    }
    
    //Get primary monitor mode
    
    const GLFWvidmode* mode;
    if((mode=glfwGetVideoMode(monitor)) == NULL)
    {
        cout<<"GLFW failed to retrieve primary monitor settings!"<<endl;
    }
    
    //Create opencv window
    
    cv::namedWindow("render", CV_WINDOW_NORMAL);
    cv::setWindowProperty("render", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    
    //Create LaptopVR engine 
    
    LaptopVR engine(mode->width, mode->height);
    Mat frame;
    
    //Init VR engine and set parameters
    
    if (!engine.init())
    {
        printf("Failed to init!\n");
        return 0;
    }
    engine.setFlipFrame(true);
    
    //calculate and set dpi
    int widthMm=0;
    glfwGetMonitorPhysicalSize(monitor, &widthMm, NULL);
    if (widthMm > 0)
    {
        float dpi = mode->width / (widthMm * 0.01 * CM_TO_INCH);
        cout << "Dpi determined: "<<dpi<<endl;
        engine.setDpi(dpi);
    }
    
    
    //Init camera from OpenCV
    
    VideoCapture cap;
    if(!cap.open(0))
    {
        cout<<"Cannot open camera!"<<endl;
        return 0;
    }
    
    //Capture-Render cycle
    
    start = clock();
    
    for(;;)
    {
        Mat vid;
        
        //Get frame
        cap >> vid;
        
        nframes++;
        
        //Get observer and render frame
        frame = engine.renderFromCamFrame(vid);
        imshow("render", frame);
        
        //engine.detectObserver(vid);
        //imshow("detect", vid);
        
        //Exit if any key pressed
        if (waitKey(1)!=-1)
        {
            break;
        }
    }
    
    double time;
    time = (clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Frame rate:"<<nframes/time<<endl;
     
     /*
    Mat plot;
    plot = plotVectors(engine.xraw, engine.yraw, engine.xfil, engine.yfil);
    imshow("x,y", plot);
    plot = plotVectors(engine.zraw, engine.yraw, engine.zfil, engine.yfil);
    imshow("z,y", plot);
    */
    
    //frame = engine.renderNextFrame(Scalar(320, 120, -1000));
    //imshow("result", frame);
    
    waitKey(0);
    
    //Terminate GLFW
    glfwTerminate();
    
}
