#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <GLFW/glfw3.h>

#include <iostream>

#include "../src/laptopVR.hpp"

using namespace std;
using namespace cv;

int main()
{
    
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
    
    for(;;)
    {
        Mat vid;
        
        //Get frame
        cap >> vid;
        
        //Get observer and render frame
        frame = engine.renderFromCamFrame(vid);
        imshow("render", frame);
        
        //Exit if any key pressed
        if (waitKey(1)!=-1)
        {
            break;
        }
    }
        
    
    //frame = engine.renderNextFrame(Scalar(320, 120, -1000));
    //imshow("result", frame);
    
    waitKey(0);
    
    //Terminate GLFW
    glfwTerminate();
    
}
