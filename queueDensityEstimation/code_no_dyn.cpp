#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
// #include "CudaFunc.h"
#include <thread>
#include <mutex>
extern "C"{
        int GetNextPhaseFLR(float* density, int cur_phase);

        int GetNextPhaseFLG(float* density, int cur_phase);

}
/*typedef struct arg_struct {
    int maxfps;
    int fpunit;
    double staticfactor;
    bool cap_on;
    bool ss_on;
    Mat ss;
    Mat frame;
    VideoCapture video;
    bool end;   
};
*/
//PLEASE NOTE: We can run this code without GPU, just comment Line 6,728
//Helpful command: `pkg-config --cflags --libs opencv`
using namespace cv;
using namespace std;

typedef struct arg_struct {
    int arg1;
    int arg2;
    double arg3;
    bool arg4;
    bool arg5;
    Mat arg6;
    Mat arg7;
    VideoCapture arg8;
    bool arg9;
};

std::mutex mu;
//usr/local/include/opencv4/opencv2/
//int ix=30,fx=200 ;
int caly ;
int calx ;
int X,Y ;
int sx,sy,dx,dy ;
double Vwidth, Vheight;
//******************************************************_______________________________________________________________________________________*******************
////PARAMETERS THAT CAN BE ALTERED FOR BETTER ACUURACY : XFIT,YFIT,DYNAMIC_TIME,STATIC_TIME,VARIANCE_FROM_AVG and inside isRED_ arguments of find_line************
//******************************************************_______________________________________________________________________________________*******************

// MUST SET BELOW PARAMETERS FIRST

double BLOB_RATIO=0.3;
int XFIT=7 ;                            //Number of vehicles that would fit in horizontal, now lets us suppose we can fit 3 cars so we may want to have XFIT atleast 3 but since we can also have bikes we can make it 4 or 5 or 6
int YFIT=7 ;                            //Similar explanation but since bikes and cars have similar length, this is not a problem
// int DYNAMIC_TIME=2 ;         //Time within which a frame would completely disappear from average, we may want to keep it small so that slow vehicles or stopped vehicles disappear quickly in seconds or FRAMEUNITS
int STATIC_TIME=30 ;            //in seconds or FRAMEUNITS
int VARIANCE_FROM_AVG =35 ; //variance from avg pixel, the variance from average pixel when it is considered an anomaly, be careful, should be between 10 to 40
int VAR =125 ;                          //variance from avg pixel, useless for now

vector<Point2f> points;
vector<Point2f> points2;
bool proj_set=false;
bool size_set=false;

Mat staticAVG,dynAVG ;
Mat staticAVG_dif,dynAVG_dif ;
double white_area_variable=0.0, white_area_variable1 = 0.0, stop_density=0.0;
ifstream infile;
ofstream outfile;

Mat prvs;
double resize_percent=0.2;

void on_mouse2( int e, int x, int y, int d, void *ptr ){

        if (e == EVENT_LBUTTONDOWN )

        {

                if(points2.size() < 3 )

                {


                        points2.push_back(Point2f(float(x),float(y)));

                        outfile << x << " "<< y <<endl;

                        cout << x << " "<< y <<endl;

                }

                else if(points2.size()==3)

                {

                        points2.push_back(Point2f(float(x),float(y)));

                        outfile << x << " "<< y <<endl;

                        cout << x << " "<< y <<endl;

                        cout << " Done, Thanks! " <<endl;

                        size_set=true;

                        // Deactivate callback

                        setMouseCallback("Display window2", NULL, NULL);

                }


        }

}

void on_mouse( int e, int x, int y, int d, void *ptr ){

        if (e == EVENT_LBUTTONDOWN )

        {

                if(points.size() < 3 )

                {


                        points.push_back(Point2f(float(x),float(y)));

                        outfile << x << " "<< y <<endl;

                        cout << x << " "<< y <<endl;

                }

                else if(points.size()==3)

                {

                        points.push_back(Point2f(float(x),float(y)));

                        outfile << x << " "<< y <<endl;

                        cout << x << " "<< y <<endl;

                        cout << " Done, Thanks! " <<endl;

                        proj_set=true;

                        // Deactivate callback

                        setMouseCallback("Display window", NULL, NULL);

                }


        }

}

void size_points(Mat src2final){



        imshow( "Display window2", src2final );

        setMouseCallback("Display window2",on_mouse2, NULL );

        while(1)

        {

                int key=waitKey(0);

                if(key==27) break;

        }

        return;

}

void proj_points(Mat ss){

        imshow( "Display window", ss );

        setMouseCallback("Display window",on_mouse, NULL );

        while(1)

        {

                int key=waitKey(0);


                if(key==27) break;

        }

        return;

}

Mat proj(Mat input_Frame,bool colour,Mat ss){


                vector<Point2f> left_image;                 // Stores 4 points(x,y) of the logo image. Here the four points are 4 corners of image.

                vector<Point2f> right_image;        // stores 4 points that the user clicks(mouse left click) in the main image.


                if(!proj_set){

                        proj_points(ss);

                }


                left_image.push_back(points[0]);

                left_image.push_back(points[1]);

                left_image.push_back(points[2]);

                left_image.push_back(points[3]);


                //Dont Change////////////////////////////////

                right_image.push_back(Point2f(float(472),float(52)));

                right_image.push_back(Point2f(float(472),float(830)));

                right_image.push_back(Point2f(float(800),float(830)));

                right_image.push_back(Point2f(float(800),float(52)));

                ////////////////////////////////////////////


                Mat H = findHomography(left_image,right_image,0 );

                Mat logoWarped;

                warpPerspective(input_Frame,logoWarped,H,Size(1280,875));

                Mat gray,src2final;

                //if(colour) cvtColor(logoWarped,gray,COLOR_BGR2GRAY); //IN COLOUR

                gray=logoWarped.clone(); //IN GRAY_SCALE

                threshold(gray,gray,0,255,THRESH_BINARY);

                logoWarped.copyTo(src2final,gray);

                if(!size_set){

                        size_points(src2final);

                }

                float p1x=points2[0].x;

                float p1y=points2[0].y;

                float p2x=points2[1].x;

                float p2y=points2[1].y;

                float p3x=points2[2].x;

                float p3y=points2[2].y;

                float p4x=points2[3].x;

                float p4y=points2[3].y;


                float a1x=(p1x+p2x)/2;

                float a1y=(p1y+p4y)/2;

                float a2x=(p3x+p4x)/2;

                float a2y=(p2y+p3y)/2;


                Mat retImage = src2final(Rect(a1x,a1y,a2x-a1x,a2y-a1y));

                //img[900:56, 389:831]

   return retImage;

}

Mat proj(Mat input_Frame,bool colour){

                // We need 4 corresponding 2D points(x,y) to calculate homography.

                vector<Point2f> left_image;                 // Stores 4 points(x,y) of the logo image. Here the four points are 4 corners of image.

                vector<Point2f> right_image;        // stores 4 points that the user clicks(mouse left click) in the main image.


                // Image containers for main and logo image

                Mat imageMain;

                Mat imageLogo=input_Frame;

                left_image.push_back(points[0]);

                left_image.push_back(points[1]);

                left_image.push_back(points[2]);

                left_image.push_back(points[3]);


                right_image.push_back(Point2f(float(472),float(52)));

                right_image.push_back(Point2f(float(472),float(830)));

                right_image.push_back(Point2f(float(800),float(830)));

                right_image.push_back(Point2f(float(800),float(52)));



                Mat H = findHomography(         left_image,right_image,0 );

                Mat logoWarped;

                // Warp the logo image to change its perspective

                warpPerspective(imageLogo,logoWarped,H,Size(1280,875));

                //showFinal(imageMain,logoWarped);


                Mat gray,gray_inv,src1final,src2final;

                if(colour) cvtColor(logoWarped,gray,COLOR_BGR2GRAY); //IN COLOUR

                else gray=logoWarped.clone(); //IN GRAY_SCALE

                threshold(gray,gray,0,255,THRESH_BINARY);

                //adaptiveThreshold(gray,gray,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,5,4);

                bitwise_not ( gray, gray_inv );

                                //imageMain.copyTo(src1final,gray_inv);

                logoWarped.copyTo(src2final,gray);

                Mat finalImage = src2final;

                float p1x=points2[0].x;

                float p1y=points2[0].y;

                float p2x=points2[1].x;

                float p2y=points2[1].y;

                float p3x=points2[2].x;

                float p3y=points2[2].y;

                float p4x=points2[3].x;

                float p4y=points2[3].y;

                float a1x=(p1x+p2x)/2;

                float a1y=(p1y+p4y)/2;

                float a2x=(p3x+p4x)/2;

                float a2y=(p2y+p3y)/2;

                Mat retImage = finalImage(Rect(a1x,a1y,a2x-a1x,a2y-a1y));

                //img[900:56, 389:831]

   return retImage;

   }

void find_line(Mat img,int xfit, int yfit, bool up,double ylimTop, double tempBlobRatio){ //note that img has to be grayscale
        //ylimtop is the ratio from top which is ylimit as upper box ;
        int ylimTopPix = (1.0-ylimTop)*img.rows;
        calx = img.cols/xfit ;
        caly = img.rows/yfit ;
        int rows = img.rows ;
        int cols = img.cols ;
        bool done=false ;
        int bw[rows-ylimTopPix][cols];
        for(int i=ylimTopPix;i<rows;i++)
                for(int j=0;j<cols;j++)
                        bw[i-ylimTopPix][j]=0;
        // Mat img2 = img.clone() ;

        if((int)img.at<uchar>(ylimTopPix,0)>150)
                bw[0][0]= 1 ;

        for(int i=1;i<cols;i++)
                if((int)img.at<uchar>(0,i)>150)
                        bw[0][i]=bw[0][i-1]+1 ;
                else
                        bw[0][i]=bw[0][i-1] ;

        for(int j=ylimTopPix+1;j<rows;j++)
                if((int)img.at<uchar>(j,0)>150)
                        bw[j-ylimTopPix][0]=bw[j-1-ylimTopPix][0]+1 ;
                else
                        bw[j-ylimTopPix][0]=bw[j-1-ylimTopPix][0] ;

        if(up){
                int y=1+ylimTopPix ; int x=1 ;
                for(;y<rows;y++){
                        for(x=1;x<cols;x++){
                                if((int)(img.at<uchar>(y,x))>150)
                                        bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1]+1;
                                else
                                        bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1];

                                if(y>=caly+ylimTopPix&&x>=calx && ((double)(((int)bw[y-ylimTopPix][x]+(int)bw[y-caly-ylimTopPix][x-calx]-(int)bw[y-caly-ylimTopPix][x]-(int)bw[y-ylimTopPix][x-calx])*1.0/(calx*caly))>tempBlobRatio))
                                        done=true ;
                                if(done)
                                        break ;
                        }
                        if(done)
                                break ;
                }
                if(!done){
                        X=0;Y=img.rows;}
                else{
                        X=x-calx ;Y=y-caly ;}


// cv::Mat A(100,100,CV_64F);

                        // Mat temp(rows, cols, CV_8UC3);
                        // std::memcpy(temp.data, bw, cols*rows*sizeof(int));
                        // imshow("bw",temp);
                        // waitKey(0);
                        // line(img2,Point(0,Y),Point(img2.cols,Y),Scalar(255),1) ;
                        // line(img2,Point(X,0),Point(X,img2.rows),Scalar(255),1) ;
        }

        else{
                int y=1+ylimTopPix ; int x=1 ;
                for(;y<rows;y++)
                        for(x=1;x<cols;x++)
                                if((int)(img.at<uchar>(y,x))>150)
                                        bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1]+1;
                                else
                                        bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1];

                for(y=rows-1;y>=caly+ylimTopPix;y--){
                        for(x=calx;x<cols;x++){
                                //cout<<"ratio:"<<(double)(((int)bw[y][x]+(int)bw[y-caly][x-calx]-(int)bw[y-caly][x]-(int)bw[y][x-calx])*1.0/(calx*caly))<<endl;
                                if((double)(((int)bw[y-ylimTopPix][x]+(int)bw[y-caly-ylimTopPix][x-calx]-(int)bw[y-caly-ylimTopPix][x]-(int)bw[y-ylimTopPix][x-calx])*1.0/(calx*caly))>tempBlobRatio)
                                        done=true ;
                                if(done)
                                        break ;
                        }
                if(done)
                        break ;
                }

                if(!done)
                        {X=0;Y=ylimTopPix;}
                else
                        {X=x-calx ;Y=y-caly ;}

                        // line(img2,Point(0,Y),Point(img2.cols,Y),Scalar(255),1) ;
                        // line(img2,Point(X,0),Point(X,img2.rows),Scalar(255),1) ;
        }
}

void find_line(Mat img,int xfit, int yfit, bool up){
        return find_line(img,xfit,yfit,up,1.00,BLOB_RATIO) ;
}

void find_line(Mat img){
        return find_line(img,XFIT,YFIT,true) ;
}

void find_line(Mat img,bool up){

                return find_line(img,XFIT,YFIT,up) ;

}

bool isRed_(Mat statMat,Mat dynMat){
        double ImgRatio = 0.4 ;
        find_line(statMat,XFIT,YFIT,false,ImgRatio,0.7) ;
        int sy=Y ;
        find_line(dynMat,XFIT,YFIT,false,ImgRatio,0.7) ;
        int dy=Y ;
        // cout<<"dy:"<<dy<<" sy:"<<sy<<" rows:"<<statMat.rows ;
        int ch1=(1.00-ImgRatio)*statMat.rows;
        bool bs=sy>ch1+2 && sy<statMat.rows-1 ;
        bool bd=dy>ch1+2 ;
        // cout<<" bs:"<<bs<<" bd:"<<bd<<endl ;
        // if(!(bs&&!bd))
                // cout<<"**********************************GREEN********************************************" ;

        return (bs&&!bd) ;
}

double white_area(Mat img){
        int rows = img.rows ;
        int cols = img.cols ;
        int countpos=0;
        int countall=0;
        for(int j=0;j<rows;j++){
                for(int i=0;i<cols;i++){
                        if((int)img.at<uchar>(j,i)>150)
                                countpos++;
                        countall++;
                }
        }
        return(double)((countpos*1.0)/(countall*1.0));

}

double white_area_dyn(Mat img){
        int rows = img.rows ;
        int cols = img.cols ;
        double countpos=0;
        int countall=0;
        for(int j=0;j<rows;j++){
                for(int i=0;i<cols;i++){
                        // if(>150)
                        int x=(int)img.at<uchar>(j,i);
                        countpos+= (x/255.0);
                        countall++;
                }
        }
        return(double)((countpos)/(countall*1.0));

}

Mat magn_norm;

int p1=0,p2=0,p3=0,p4=0;

int proj_rows, proj_cols;

void runDenseFlow(Mat frame){
        // t7 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
        auto t = chrono::high_resolution_clock::now();
        Mat frame2 = proj(frame,0);

        Mat dst,next;
        // cout<<"frame2 size: "<<frame2.size()<<endl;
        // resize(frame2, frame2, Size((int)(proj_cols*resize_percent),(int)proj_rows*resize_percent));
        resize(frame2, frame2, Size(),resize_percent,resize_percent,INTER_CUBIC);
        // cout<<"frame2 updated size: "<<frame2.size()<<endl;
        p1 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
        t = chrono::high_resolution_clock::now();
    if (frame2.empty())
        return;
    cvtColor(frame2, next, COLOR_BGR2GRAY);
    prvs= proj(prvs,0);
    resize(prvs, prvs, Size(),resize_percent,resize_percent,INTER_CUBIC);
    Mat flow(prvs.size(), CV_32FC2);
    p2 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
        t = chrono::high_resolution_clock::now();
    // cout<<"next.size(): "<<next.size()<<" and prvs.size(): "<<prvs.size()<<"prvs.channels(): "<<prvs.channels()<<"next.channels(): "<<next.channels()<<endl;
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    // cout<<"next.size(): "<<next.size()<<" and prvs.size(): "<<prvs.size()<<endl;
    // visualization
    p3 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
        t = chrono::high_resolution_clock::now();
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);

    magn_norm.setTo(2, magn_norm < 0.25);
    magn_norm.setTo(85,magn_norm<=0.4);
    magn_norm.setTo(255, magn_norm <=1);
    // magn_norm.setTo(255, magn_norm <=1);
    // cout<<"magn_norm: "<<endl<<(magn_norm)<<endl;
    // magn_norm.setTo(0,magn_norm==2);
    // magn_norm.setTo(125,magn_norm>=0.3 && magn_norm<0.5);

    // magn_norm = magn_norm*255;
    p4 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
        t = chrono::high_resolution_clock::now();


        // cout<<"ps: " <<p1<<" "<<p2<<" "<<p3<<" "<<p4<<endl;
    // // std::transform(magnitude.begin<double>(),magnitude.end<double>(),magnitude.begin<double>(),mat_transform_fnc);
    // angle *= ((1.f / 360.f) * (180.f / 255.f));
    // //build hsv image
    // Mat _hsv[3], hsv, hsv8, bgr;
    // _hsv[0] = angle;
    // _hsv[1] = Mat::ones(angle.size(), CV_32F);
    // _hsv[2] = magn_norm;
    // merge(_hsv, 3, hsv);
    // hsv.convertTo(hsv8, CV_8U, 255.0);
    // cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    // Mat temp1 = bgr+frame2;
    // resize(temp1, temp1, Size(),1/resize_percent,1/resize_percent,INTER_CUBIC);
    // imshow("frame2", temp1);
    // int keyboard = waitKey(1);
    // if (keyboard == 'q' || keyboard == 27)
        // break;

    // cout<<
        // cout<<"magn_norm.size(): "<<magn_norm.size()<<" and dynAVG.size(): "<<dynAVG.size()<<"dynAVG.channels(): "<<dynAVG.channels()<<"magn_norm.channels(): "<<magn_norm.channels()<<endl;



        // cvtColor(magn_norm, dynAVG, COLOR_BGR2GRAY);

        // cout << "magn_norm = " << endl << " "  << magn_norm << endl << endl;
        // cout << "dynAVG = " << endl << " "  << dynAVG << endl << endl;

        // cout<<"dynavg.shape: "<<dynAVG.rows<<","<<dynAVG.cols<<" and dynavg.channels: "<<dynAVG.channels()<<endl;

    // resize(frame2, dynAVG, Size(frame.cols,frame.rows));



    // prvs = next;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


//void func_fps(int MAXFPS_LIM, int FPUNIT, double STATIC_FACTOR,bool CAP_ON, bool SS_ON, Mat SS,Mat frame,VideoCapture Video_cap,bool isEND){
  
 void *func_fps(void *arguments){
                   int t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0,t8=0,t9=0;
                    int loop =0;
                    struct arg_struct *args = (struct arg_struct *)arguments;
                    int MAXFPS_LIM=args->arg1;
                    int FPUNIT= args->arg2;
                    double STATIC_FACTOR=args->arg3;
                    bool CAP_ON=args->arg4;
                   bool SS_ON= args-> arg5;
                   Mat SS= args->arg6;
                   Mat frame= args-> arg7;
                   VideoCapture Video_cap= args->arg8;
                   bool isEND= args->arg9;
                    Mat avgframe,ErodeDilateMat,MedianBlurMat,difframe,difframe_2,difframe_3,
                        difdisplayframe,difdisplayframe_2,difframe_result,temp2frame,blackwhiteframe,tempframe,
                        projDYN,projSTAT,projCOL,staticINIT;
                       cout<<"Entered the function fps for a thread"<<endl;
                      float totaltime=0.0;
                        int loop_counter=0;
                       constexpr int ITERATIONS = 120;
                        bool isRED=false ;
                        isEND= false ;
                        double dropratio = FPUNIT*1.0/MAXFPS_LIM ;
                        //auto startloop = chrono::high_resolution_clock::now() ;
                        auto time1      = chrono::high_resolution_clock::now();
                        int i =0;
             //   for(int i=0;i<1;i++){
                       cout<<"Enetered this loop"<<endl;  
                        while(i<ITERATIONS){
                                 mu.lock();
                                 auto startloop = chrono::high_resolution_clock::now() ;
                                auto t = chrono::high_resolution_clock::now() ;

                                if(loop_counter%5==0){
                                        float densities[4] = {0.5,0.33,0.33,0.33};
                                        t = chrono::high_resolution_clock::now() ;
                                  //      cout<<"GetNextPhaseFLR: "<<GetNextPhaseFLR(densities, 1)<<endl;
                                        t1 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                        t = chrono::high_resolution_clock::now() ;
                                    //    cout<<"GetNextPhaseFLG: "<<GetNextPhaseFLG(densities,1)<<endl;
                                        t2 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                }
                                else{
                                        int counter=0;                  //counter goes upto fps
                                        int success_counter=1 ; //successcounter goes upto MAXFPS_LIM



                                        while(counter<FPUNIT){

                                                bool success = Video_cap.read(frame) ;
                                                if(!success) {
                                                        isEND=true ;
                                                        cout<<"Cannot read frame anymore"<<endl;
                                                        break ;
                                                }
                                                // t1 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());

                                                counter++ ;

                                                cvtColor(frame, blackwhiteframe, COLOR_BGR2GRAY); //TO black and white

                                                if(CAP_ON&&counter!=floor(success_counter*dropratio)){
                                                  //      cout<<"skipping frame :"<<counter<<endl;
                                                        prvs = blackwhiteframe.clone();
                                                        continue ;
                                                }


                                                t = chrono::high_resolution_clock::now();

                                                if(counter==FPUNIT)
                                                runDenseFlow(frame);
                                                prvs = blackwhiteframe.clone();
                                                // }
                                                // else{
                                                        // prvs = proj(blackwhiteframe,0);
                                                        // resize(prvs, prvs, Size(),resize_percent,resize_percent,INTER_CUBIC);
                                                // }
                                                t7 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                                t = chrono::high_resolution_clock::now();

                                                // if(GPU_ON)
                                                //      tempframes[success_counter-1]=blackwhiteframe.clone();
                                                // else
                                                //      tempframesCPUV2[success_counter-1]=blackwhiteframe.clone();
                                                t8 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                                t = chrono::high_resolution_clock::now();
                                                success_counter++;

                                                if(success_counter>MAXFPS_LIM+1)
                                                    //    cout<<"ERERERERERERERERERERERERERERERRERERERERERERERERERERERERERERERERERERERERERERERERERERERER ....... CONTACT THE ADMIN! "<<endl ;

                                                if(!isRED) {
                                                        staticAVG=STATIC_FACTOR*staticAVG+(1-STATIC_FACTOR)*blackwhiteframe;
                                                        // cout<<"------------------static_changes---------------------------"<<endl;
                                                }
                                                t9 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                                t = chrono::high_resolution_clock::now();
                                        }

                                        t1 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                        t = chrono::high_resolution_clock::now();

                                        // cout<<"SUCCESSCOUNTER = "<<success_counter<<endl ;
                                        // cout<<"Loop going with height: "<<height<<" and width: "<<width<<endl ;

                                        resize(magn_norm, dynAVG, Size(frame.cols,frame.rows));
                                        // imshow("magn_norm",magn_norm);
                                        // imshow("dynAVG", dynAVG);
                                        // waitKey(0);

                                        // cout<<"dynAVG size: "<<dynAVG.size()<<" and frame.size(): "<<frame.size()<<endl;

                                        // char sum_arr[height*width] ;
                                        // cout<<"instantiated array 1"<<endl;

                                        // char sum_array[height][width];

                                        //We want to receive sum_array from this part, Either GPU or CPU
                                        // if(GPU_ON){
                                        //      cout<<"inside gpu_on"<<endl;

                                        //      for(int f=0;f<framefactor;f++){
                                        //              // cout<<"framefactor: "<<framefactor<<endl;
                                        //              for(int r=0;r<square;r++){
                                        //                      // cout<<"r: "<<r<<endl;
                                        //                      for(int y=0;y<height;y++){
                                        //                              for(int x=0;x<width;x++){
                                        //                                      // if(y==1079)cout<<"x: "<<x<<endl;
                                        //                                      converted_arr[f][r*height*width+y*width+x]=((char)(tempframes[f*square+r].at<uchar>(y,x)));
                                        //                              }
                                        //                      }
                                        //              }
                                        //      }

                                        //      cout<<"done here"<<endl;

                                        //      auto st = chrono::high_resolution_clock::now() ;
                                        //      cout<<"Gpu wfunction called . . ." ;
                                        //      // AvgCalGpu(height,width,3,framefactor,converted_arr,sum_arr) ;
                                        //      cout<<" . . . Gpu function has returned"<<endl ;
                                        //      cout<<"Time taken by gpu function "<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-st).count()/1000.0<<endl;

                                        //      for(int i=0;i<height;i++)
                                        //              for(int j=0;j<width;j++)
                                        //                      sum_array[i][j]=sum_arr[(i*width)+j];
                                        // }
                                        // else{
                                        //      int tempsum=0 ;
                                        //      for(int y=0;y<height;y++)
                                        //              for(int x=0;x<width;x++){
                                        //                      tempsum=0;

                                        //                      for(int i=0;i<MAXFPS_LIM;i++)
                                        //                                      tempsum+=((int)(tempframesCPUV2[i].at<uchar>(y,x))) ;

                                        //                      tempsum/=MAXFPS_LIM ;
                                        //                      sum_array[y][x] = (char)tempsum ;
                                        //              }
                                        // }

                                        // Mat A(height,width,CV_8UC1,sum_array);
                                        // std::memcpy(A.data,sum_array,height*width*sizeof(char));
                                        // //imwrite("Avgphoto.jpeg",A) ;
                                        // Mat Aframe ;
                                        // A.convertTo(Aframe,CV_64FC1) ;
                                        // Mat Bframe=Aframe.clone();

                                        // sum_tempframes-=tempframes_total[loop_counter%DYNAMIC_TIME]/DYNAMIC_TIME;
                                        // tempframes_total[loop_counter%DYNAMIC_TIME]=Bframe;
                                        // sum_tempframes+=Bframe/DYNAMIC_TIME;
                                        // //imwrite("sumframes.jpeg",sum_tempframes);
                                        // sum_tempframes.convertTo(dynAVG,CV_8UC1);
                                        //imwrite("avg.jpeg",dynAVG) ;

                                        ///DYNAVG CODE OVER.... dynAVG is not correct
                                        // cout<<"starting this 1"<<endl;

                                        Mat temp1 = cv::abs(blackwhiteframe-staticAVG);
                                        // cout<<"starting this 2"<<endl;

                                        // cout<<"blackwhiteframe.size(): "<<blackwhiteframe.size()<<" and dynAVG.size(): "<<dynAVG.size()<<"dynAVG.channels(): "<<dynAVG.channels()<<"blackwhiteframe.channels(): "<<blackwhiteframe.channels()<<endl;

                                        // cout<<"blackwhiteframe.type(): "<<type2str(blackwhiteframe.type())<<" and dynAVG.type(): "<<type2str(dynAVG.type())<<endl;
                                        dynAVG.convertTo(dynAVG, CV_8UC1);
                                        // Mat temp2 = cv::abs(blackwhiteframe-dynAVG);

                                        t2 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                        t = chrono::high_resolution_clock::now();


                                        // cout<<"starting this 3"<<endl;
                                        threshold(temp1,staticAVG_dif,VARIANCE_FROM_AVG,255,THRESH_BINARY) ;
                                        // cout<<"starting this 4"<<endl;
                                        // threshold(dynAVG,dynAVG_dif,VARIANCE_FROM_AVG,255,THRESH_BINARY) ;
                                        // cout<<"starting this 5"<<endl;

                                        t3 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                        t = chrono::high_resolution_clock::now();

                                        //imwrite("dynavgdiff.jpeg",dynAVG_dif);

                                        // bool isRED = isRed(proj(staticAVG_dif,false,SS),proj(dynAVG_dif,false,SS));
                                        if(SS_ON){
                                                projDYN = proj(dynAVG,0,SS);
                                                projSTAT = proj(staticAVG_dif,0,SS) ;
                                        }else{
                                                projDYN = proj(dynAVG,0);
                                                projSTAT = proj(staticAVG_dif,0) ;
                                        }

                                        t4 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                        t = chrono::high_resolution_clock::now();

                                        isRED = isRed_(projSTAT,projDYN);
                                        white_area_variable = white_area(projSTAT);
                                        white_area_variable1 = white_area_dyn(projDYN);
                                        white_area_variable1 = (white_area_variable1>white_area_variable)?white_area_variable:white_area_variable1;
                                        stop_density = white_area_variable - white_area_variable1;

                                        find_line(projSTAT) ;
                                        line(projSTAT,Point(0,Y),Point(projSTAT.cols,Y),Scalar(255),1) ;
                                        sy=Y ;
                                        find_line(projDYN) ;
                                        dy=Y ;

                                        t5 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                        t = chrono::high_resolution_clock::now();


                                        //imshow("Original",blackwhiteframe) ;
                                        //imshow("Average",dynAVG) ;
                                        //imshow("Difframe",dynAVG_dif) ;

                                        //****************************************   U N C O M M E N T   T H E   F O L L O W I N G   F O R    D E  B U G G I N G   *********************************************
                                //      imshow("Original_Proj",proj(frame,1)) ;

                                //      Mat stop_dens = projSTAT - projDYN;
                                //      stop_dens.setTo(0,stop_dens<0);
//                                      imshow("STOP DENSITY", stop_dens);
//
//                                      imshow("DYNAMIC",projDYN) ;
//
//                                      imshow("STATIC",projSTAT) ;  // ANSWER FRAME
//                                      waitKey(0);


                                        // cout<<"Static_Y: "<<sy<<","<<"Dynamic_Y: "<<dy<<endl ;

                                        //writing part
                                        string output_ans = to_string(loop)+":"+to_string(white_area_variable)+","+to_string(white_area_variable1)+","+to_string(sy*1.0/projSTAT.rows) ;
                                        outfile<<output_ans<<endl ;
                                        //cout<<output_ans<<endl;
                                }
                                        //cout<<loop++<<endl;

                                // while(WAIT_ON){
                        //      while(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-startloop).count()/(loop_counter*1.0) < 1000000){};

                                // }

                                // t6 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
                                // t = chrono::high_resolution_clock::now();
                               ///auto t_end = std::chrono::high_resolution_clock::now();
                                  //  float ms = std::chrono::duration<float, std::micro>(t_end - startloop).count();

                               // cout<<t1<<" "<<t2<<endl;//" "<<t3<<" "<<t4<<" "<<t5<<" "<<t6<<" "<<t7<<" "<<t8<<" "<<t9<<endl;
                           //double timetaken= chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-startloop).count();
                                //cout<<"The final time taken is "<<ms/1000000<<endl;
                                // if(waitKey(1)==27) break ;
        //                      cout<<"total time taken: "<<fixed<<setprecision(4)<<(double)
//(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-startloop).count())<<endl;
                                loop_counter++;

                                if(loop_counter==5){
                                        // waitKey(0);
                                        time1   = chrono::high_resolution_clock::now();

                                }
                                else if(loop_counter==105){
                                 //       cout<<"time for 100 loops: "<<(double)(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-time1).count())<<endl;
                                        // break;
                                }
                          i++;

                         auto t_end = std::chrono::high_resolution_clock::now();
                         float micro = std::chrono::duration<float, std::micro>(t_end - startloop).count();
                         float second=micro/1000000;
                          totaltime=totaltime+second;
                               // cout<<t1<<" "<<t2<<endl;//" "<<t3<<" "<<t4<<" "<<t5<<" "<<t6<<" "<<t7<<" "<<t8<<" "<<t9<<en$
                           //double timetaken= chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now$
                              //  cout<<"The final time taken is "<<ms/1000000<<endl;

                       //   isEND=true;
                          mu.unlock();
                        }


                       cout<<"The total time is given by "<<totaltime/ITERATIONS<<endl;
                        //imwrite("Size_det.jpg",frame) ;
                        cout<<"I am out of the loop,Press Esc to exit!"<<endl ;
//                        cout<<"Total time to run code:"<<chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now()-startcode).count()<<endl;

                        if(waitKey(1000)==27) {
                                infile.close();
                                outfile.close();
                                return 0;
                        }
           // }




}







int main(int argc, char *argv[]){


        int t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0,t8=0,t9=0;
        // auto t;

        auto startcode = chrono::high_resolution_clock::now();

        //argv[1] is video name
        //argv[2] is whether or not you have to set four values to configure the projection thing ;
        //argv[3] is whether or not GPU is on
        //argv[4] is if you want to limit the fps
        //argv[5] is whether you want data to acquire as soon as it is processed or wait for whole one second ;
                double STATIC_FACTOR=0.0 ;                      //This is calculated as the number by which a frame will reduce to half its importance in STATIC_TIME
        STATIC_FACTOR = 0.9 ; //chnageeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        cout<<"STATIC_FACTOR"<<STATIC_FACTOR<<endl ;

        double li[] = {0.998};
//      for(int i=0;i<1;i++){
                STATIC_FACTOR = li[0];
                cout<<"doing for STATIC_FACTOR: "<<STATIC_FACTOR<<endl;
                if(argc<2) cout<<"Video name missing!"<<endl ;
                else{
                        bool GPU_ON = false ;
                        bool SS_ON = false ;
                        bool WAIT_ON = false ;
                        bool CAP_ON = true ;
                        int MAXFPS_LIM ;
                        string tempString="" ;
                        string io=argv[1];
                        io+=".txt";
                        if(argc>2){
                                tempString=argv[2] ;
                                SS_ON = stoi(tempString)!=0;
                                if(SS_ON)
                                        outfile.open(io);
                                else
                                        infile.open(io);
                        }
                        else{
                                infile.open(io);
                        }
                        if(!SS_ON){
                                float fx,fy;
                                for(int i=0;i<4;i++){
                                        infile>>fx;
                                        infile>>fy;
                                        points.push_back(Point2f(fx,fy));
                                }
                                for(int i=0;i<4;i++){
                                        infile>>fx;
                                        infile>>fy;
                                        points2.push_back(Point2f(fx,fy));
                                }
                        }
                        if(argc>3){
                                tempString=argv[3] ;
                                // GPU_ON = stoi(tempString)!=0 ;
                        }
                        if(argc>5){
                                tempString=argv[5] ;
                                WAIT_ON = stoi(tempString)!=0 ;
                        }


                        //OPen a out file to write
                        ofstream outfile;
                        string inname = argv[1] ;
                        string outname = "results_gpu_final_"+to_string(STATIC_FACTOR)+""+".txt" ;
                        outfile.open(outname.c_str());
                        //Video
                        string vid_name = argv[1] ;
                        VideoCapture Video_cap(vid_name) ;
                        if(!Video_cap.isOpened()){
                                cout << "Error opening video stream or file" << endl;
                                return -1;
                        }

                        //Initialisation
                        Vwidth = Video_cap.get(CAP_PROP_FRAME_WIDTH) ;
                        Vheight = Video_cap.get(CAP_PROP_FRAME_HEIGHT);
                        double fps = round(Video_cap.get(CAP_PROP_FPS)) ;
                        int FPUNIT = (int) fps ;                                //We consider a certain number of frames as new FPUNIT rather than FPS because it is more generic
                         // namedWindow("Average",WINDOW_NORMAL) ;
                        // resizeWindow("Average", Vwidth/3,Vheight/3) ;
                         // namedWindow("Original",WINDOW_NORMAL) ;
                        // resizeWindow("Original", Vwidth/3,Vheight/3) ;
                         // namedWindow("Difframe",WINDOW_NORMAL) ;
                        // resizeWindow("Difframe", Vwidth/3,Vheight/3) ;

                        //MAXFPS_LIM is the max numebr of frames that will be used out of that FPUNIT, note that when gpu is on the MAXFPS_LIM will be on automatically because we work only in terms of perfect squares or their multiples
                        MAXFPS_LIM = 15 ;
                        if(GPU_ON){
                                CAP_ON=true ;
                                // MAXFPS_LIM = FPUNIT ;
                        }
                        if(argc>4){
                                CAP_ON=true;
                                tempString = argv[4] ;
                                if(stoi(tempString)>0&&stoi(tempString)<FPUNIT){
                                        CAP_ON=true ;
                                        MAXFPS_LIM = stoi(tempString) ;
                                }
                        }

                        cout<<"Width: "<<Vwidth<<endl ;
                        cout<<"Length: "<<Vheight<<endl ;
                        cout<<"FPS: "<<fps<<endl ;
                        cout<<"GPU_ON: "<<GPU_ON<<endl ;
                        cout<<"CAP_ON: "<<CAP_ON<<endl ;
                        cout<<"WAIT_ON: "<<WAIT_ON<<endl ;
                        Mat frame(Size(Vwidth,Vheight),CV_8UC3,Scalar(0)),sumframe(frame.size(),CV_64FC3,Scalar(0)) ;
                        Mat avgframe,ErodeDilateMat,MedianBlurMat,difframe,difframe_2,difframe_3,
                        difdisplayframe,difdisplayframe_2,difframe_result,temp2frame,blackwhiteframe,tempframe,
                        projDYN,projSTAT,projCOL,staticINIT,SS ;

                        //INITIALISATION
                        int height = frame.rows ;
                        int width = frame.cols ;
                        imwrite("intialbkg.jpg",frame) ;
                        // cout<<"done here 3"<<endl;
                        cvtColor(frame, blackwhiteframe, COLOR_BGR2GRAY);
                        // cout<<"done here 4"<<endl;

                        staticAVG = imread("background.png", IMREAD_UNCHANGED); ; //STATICAVG of b&w frame of original, no projection
                        cvtColor(staticAVG, staticAVG, COLOR_BGR2GRAY);

                        // dynAVG=blackwhiteframe.clone();              //DYNAVG of b&w frame of original, no projection
                        // cout<<"done here 5"<<endl;

                        //for projection selection///////////////////////////////////
                        if(SS_ON){
                                SS=staticAVG.clone();
                                SS=proj(SS,true,SS);
                        }



                        Mat temp_f = proj(frame,0);
                        proj_rows = temp_f.rows;
                        proj_cols = temp_f.cols;

                        cout<<"proj_cols: "<<proj_cols<<" and proj_rows: "<<proj_rows<<endl;

                        cout<<"done here 1"<<endl;





                        //read videos/skip videos for initial one second ;
                        for(int i=0;i<FPUNIT;i++){
                                if(!Video_cap.read(frame)){
                                cout<<"CANNOT READ FRAMES, TECHNICAL ERROR, PLEASE CHECK!";
                                return -1 ;
                                }
                                else{
                                        cvtColor(frame, prvs, COLOR_BGR2GRAY);

                                        // prvs = proj(prvs,0);
                                        // resize(prvs, prvs, Size(),resize_percent,resize_percent,INTER_CUBIC);
                                }
                        }

                        // cout<<"done here 2"<<endl;





                        //***********************INITIALISATION FOR GPU PART BEGINS*********************************
                        //GPU_ONLY
                        //framefactor is just a number less than equal to root which can be used to send data or take average in multiple of root*root.
                        int framefactor=1;
                        int root=3;
                        int square=root*root;
                        int close = (int) FPUNIT ;

                        // cout<<"done here 4"<<endl;

                        if(GPU_ON){
                                for(int r=1;r<5;r++)
                                        for(int f=1;f<r+1;f++)
                                                if(MAXFPS_LIM - r*r*f>=0&&MAXFPS_LIM - r*r*f<close){
                                                        close = MAXFPS_LIM - r*r*f ;
                                                        root=r ;
                                                        square= r*r ;
                                                        framefactor=f ;
                                                }
                                MAXFPS_LIM = square*framefactor ;
                        }
                        cout<<"MAX_FPS: "<<MAXFPS_LIM<<endl ;
                        cout<<"SQUARE: "<<square<<endl ;
                        cout<<"FRAMEFACTOR: "<<framefactor<<endl ;
                        //ALWAYS
                        //tempframes are all the frames which are sent for their dynamic average to be considered
                        //int loop=0;
                        Mat tempframesCPUV2[MAXFPS_LIM] ;
                        Mat tempframes[square*framefactor];
                        // Mat tempframes_total[DYNAMIC_TIME];

                        // for(int i=0;i<DYNAMIC_TIME;i++){
                        //      Mat local ;
                        //      (blackwhiteframe).convertTo(local,CV_64FC1) ;
                        //      tempframes_total[i]=local ;
                        // }

                        // Mat sum_tempframes;//sum of the DYNAMIC_TIME(default=6) number of average frames;
                        //Mat avg_tempframes;//avg of the DYNAMIC_TIME(default=6) number of average frames;
                        // for(int i=0;i<DYNAMIC_TIME;i++)
                        //      if(i==0)
                        //              sum_tempframes=tempframes_total[i]/DYNAMIC_TIME;
                        //      else
                        //              sum_tempframes+=tempframes_total[i]/DYNAMIC_TIME;

                        // cout<<"Sumtempframes "<<sum_tempframes.size()<<" "<<sum_tempframes.channels()<<" " ;
                        // sum_tempframes.convertTo(dynAVG,CV_8UC1,1.0/DYNAMIC_TIME);


                        // imwrite("firstimg.jpeg",dynAVG) ;
                        // char *converted_arr[framefactor];

                        // //GPU_ONLY
                        // for(int i=0;i<framefactor;i++)
                        //      converted_arr[i]=(char *)malloc(square*height*width) ;
                        //***********************INITIALISATION FOR GPU PART ENDS*********************************


   //     	func_fps(MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap);
         pthread_t tid,tid1,tid2,tid3,tid4,tid5,tid6,tid7;
       struct arg_struct args;
       args.arg1 = MAXFPS_LIM;
       args.arg2 = FPUNIT;
        args.arg3 = STATIC_FACTOR;
       args.arg4 = CAP_ON;
        args.arg5 = SS_ON;
       args.arg6 = SS;
        args.arg7 = frame;
       args.arg8 = Video_cap;
       args.arg9 =false;
//        pthread_join(tid,NULL);
// pthread_create(&tid,NULL,func_fps,&args);

  cpu_set_t cpuset,cpuset1,cpuset2,cpuset3,cpuset4,cpuset5,cpuset6,cpuset7;

  //the CPU we whant to use
  int cpu = 0;
  int cpu1 = 1;
  int cpu2=2;
   int cpu3=3;
  int cpu4=4;
  int cpu5=5;
  int cpu6=0;
   int cpu7=1;



//Thread1 running
  CPU_ZERO(&cpuset);       //clears the cpuset
  CPU_SET( cpu , &cpuset); //set CPU 2 on cpuset

pthread_create(&tid,NULL,func_fps,&args);

//Thread2 
CPU_ZERO(&cpuset1);       //clears the cpuset
  CPU_SET( cpu1 , &cpuset1); //set CPU 2 on cpuset

pthread_create(&tid1,NULL,func_fps,&args);

//Thread 3
CPU_ZERO(&cpuset2);       //clears the cpuset
 CPU_SET( cpu2 , &cpuset2); //set CPU 2 on cpuset

pthread_create(&tid2,NULL,func_fps,&args);


//Thread 4
CPU_ZERO(&cpuset3);       //clears the cpuset
 CPU_SET( cpu3 , &cpuset3); //set CPU 2 on cpuset

pthread_create(&tid3,NULL,func_fps,&args);


//Thread 5
CPU_ZERO(&cpuset4);       //clears the cpuset
 CPU_SET( cpu4 , &cpuset4); //set CPU 2 on cpuset

pthread_create(&tid4,NULL,func_fps,&args);


//Thread 6
CPU_ZERO(&cpuset5);       //clears the cpuset
CPU_SET( cpu5 , &cpuset5); //set CPU 2 on cpuset

pthread_create(&tid5,NULL,func_fps,&args);

//Thread 7
CPU_ZERO(&cpuset6);       //clears the cpuset
CPU_SET( cpu6 , &cpuset6); //set CPU 2 on cpuset

pthread_create(&tid6,NULL,func_fps,&args);

//Thread 8
CPU_ZERO(&cpuset7);       //clears the cpuset
CPU_SET( cpu7 , &cpuset7); //set CPU 2 on cpuset

pthread_create(&tid7,NULL,func_fps,&args);




   pthread_setaffinity_np(tid,sizeof(cpuset),&cpuset);
    pthread_setaffinity_np(tid1,sizeof(cpuset1),&cpuset1);
    pthread_setaffinity_np(tid2,sizeof(cpuset2),&cpuset2);
 pthread_setaffinity_np(tid3,sizeof(cpuset3),&cpuset3);
  pthread_setaffinity_np(tid4,sizeof(cpuset4),&cpuset4);
pthread_setaffinity_np(tid5,sizeof(cpuset5),&cpuset5);
pthread_setaffinity_np(tid6,sizeof(cpuset6),&cpuset6);
pthread_setaffinity_np(tid7,sizeof(cpuset7),&cpuset7);

 
   // int s = pthread_getaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
    //cout<<"The core is "<<s<<endl;
   pthread_join(tid,NULL);              
   pthread_join(tid1,NULL);
   pthread_join(tid2,NULL);
  pthread_join(tid3,NULL);

   pthread_join(tid4,NULL);
   pthread_join(tid5,NULL);
   pthread_join(tid6,NULL);
   pthread_join(tid7,NULL);

//      thread tid(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
     //  thread tid1(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
     //  thread tid2(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
    //  thread tid3(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
//thread tid4(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
  //   thread tid5(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
   // thread tid6(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
//thread tid7(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);

//thread tid8(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
  //  thread tid9(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
  // thread tid10(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
//thread tid11(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);

 //thread tid12(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
   // thread tid13(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
   //thread tid14(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
//thread tid15(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);

//thread tid16(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
  //   thread tid17(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
   // thread tid18(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
//thread tid19(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);

//thread tid20(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
  //    thread tid21(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
    // thread tid22(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
 //thread tid23(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);


//thread tid24(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
  //     thread tid25(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
    //    thread tid26(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);
 //thread tid27(func_fps,MAXFPS_LIM,FPUNIT, STATIC_FACTOR,CAP_ON,SS_ON,SS,frame,Video_cap,false);


      // tid.join();
     // tid1.join();
    //  tid2.join();
    //  tid3.join();
   //  tid4.join();
    // tid5.join();
   // tid6.join();
     // tid7.join();
   // tid8.join();
   //  tid9.join();
  // tid10.join();
 //  tid11.join();
   //  tid12.join();
  //  tid13.join();
   // tid14.join();
  // tid15.join();
  // tid16.join();
  // tid17.join();
 //  tid18.join();
 //  tid19.join();
 //  tid20.join();
 //  tid21.join();
 //  tid22.join();
  // tid23.join();
   //   tid24.join();
     // tid25.join();
     // tid26.join();
     // tid27.join();


 
//                         int loop_counter=0;
//                         bool isRED=false ;
//                         bool isEND= false ;
//                         double dropratio = FPUNIT*1.0/MAXFPS_LIM ;
//                         auto startloop = chrono::high_resolution_clock::now() ;
//                         auto time1      = chrono::high_resolution_clock::now();

//                         while(!isEND){
//                                  // auto startloop = chrono::high_resolution_clock::now() ;
//                                 auto t = chrono::high_resolution_clock::now() ;

//                                 if(loop_counter%5==0){
//                                         float densities[4] = {0.5,0.33,0.33,0.33};
//                                         t = chrono::high_resolution_clock::now() ;
//                                         cout<<"GetNextPhaseFLR: "<<GetNextPhaseFLR(densities, 1)<<endl;
//                                         t1 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                         t = chrono::high_resolution_clock::now() ;
//                                         cout<<"GetNextPhaseFLG: "<<GetNextPhaseFLG(densities,1)<<endl;
//                                         t2 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                 }
//                                 else{
//                                         int counter=0;                  //counter goes upto fps
//                                         int success_counter=1 ; //successcounter goes upto MAXFPS_LIM



//                                         while(counter<FPUNIT){

//                                                 bool success = Video_cap.read(frame) ;
//                                                 if(!success) {
//                                                         isEND=true ;
//                                                         cout<<"Cannot read frame anymore"<<endl;
//                                                         break ;
//                                                 }

//                                                 // t1 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());

//                                                 counter++ ;

//                                                 cvtColor(frame, blackwhiteframe, COLOR_BGR2GRAY); //TO black and white

//                                                 if(CAP_ON&&counter!=floor(success_counter*dropratio)){
//                                                         cout<<"skipping frame :"<<counter<<endl;
//                                                         prvs = blackwhiteframe.clone();
//                                                         continue ;
//                                                 }


//                                                 t = chrono::high_resolution_clock::now();

//                                                 if(counter==FPUNIT)
//                                                 runDenseFlow(frame);
//                                                 prvs = blackwhiteframe.clone();
//                                                 // }
//                                                 // else{
//                                                         // prvs = proj(blackwhiteframe,0);
//                                                         // resize(prvs, prvs, Size(),resize_percent,resize_percent,INTER_CUBIC);
//                                                 // }
//                                                 t7 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                                 t = chrono::high_resolution_clock::now();

//                                                 // if(GPU_ON)
//                                                 //      tempframes[success_counter-1]=blackwhiteframe.clone();
//                                                 // else
//                                                 //      tempframesCPUV2[success_counter-1]=blackwhiteframe.clone();
//                                                 t8 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                                 t = chrono::high_resolution_clock::now();
//                                                 success_counter++;

//                                                 if(success_counter>MAXFPS_LIM+1)
//                                                         cout<<"ERERERERERERERERERERERERERERERRERERERERERERERERERERERERERERERERERERERERERERERERERERERER ....... CONTACT THE ADMIN! "<<endl ;

//                                                 if(!isRED) {
//                                                         staticAVG=STATIC_FACTOR*staticAVG+(1-STATIC_FACTOR)*blackwhiteframe;
//                                                         // cout<<"------------------static_changes---------------------------"<<endl;
//                                                 }
//                                                 t9 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                                 t = chrono::high_resolution_clock::now();
//                                         }

//                                         t1 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                         t = chrono::high_resolution_clock::now();

//                                         // cout<<"SUCCESSCOUNTER = "<<success_counter<<endl ;
//                                         // cout<<"Loop going with height: "<<height<<" and width: "<<width<<endl ;

//                                         resize(magn_norm, dynAVG, Size(frame.cols,frame.rows));
//                                         // imshow("magn_norm",magn_norm);
//                                         // imshow("dynAVG", dynAVG);
//                                         // waitKey(0);

//                                         // cout<<"dynAVG size: "<<dynAVG.size()<<" and frame.size(): "<<frame.size()<<endl;

//                                         // char sum_arr[height*width] ;
//                                         // cout<<"instantiated array 1"<<endl;

//                                         // char sum_array[height][width];

//                                         //We want to receive sum_array from this part, Either GPU or CPU
//                                         // if(GPU_ON){
//                                         //      cout<<"inside gpu_on"<<endl;

//                                         //      for(int f=0;f<framefactor;f++){
//                                         //              // cout<<"framefactor: "<<framefactor<<endl;
//                                         //              for(int r=0;r<square;r++){
//                                         //                      // cout<<"r: "<<r<<endl;
//                                         //                      for(int y=0;y<height;y++){
//                                         //                              for(int x=0;x<width;x++){
//                                         //                                      // if(y==1079)cout<<"x: "<<x<<endl;
//                                         //                                      converted_arr[f][r*height*width+y*width+x]=((char)(tempframes[f*square+r].at<uchar>(y,x)));
//                                         //                              }
//                                         //                      }
//                                         //              }
//                                         //      }

//                                         //      cout<<"done here"<<endl;

//                                         //      auto st = chrono::high_resolution_clock::now() ;
//                                         //      cout<<"Gpu function called . . ." ;
//                                         //      // AvgCalGpu(height,width,3,framefactor,converted_arr,sum_arr) ;
//                                         //      cout<<" . . . Gpu function has returned"<<endl ;
//                                         //      cout<<"Time taken by gpu function "<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-st).count()/1000.0<<endl;

//                                         //      for(int i=0;i<height;i++)
//                                         //              for(int j=0;j<width;j++)
//                                         //                      sum_array[i][j]=sum_arr[(i*width)+j];
//                                         // }
//                                         // else{
//                                         //      int tempsum=0 ;
//                                         //      for(int y=0;y<height;y++)
//                                         //              for(int x=0;x<width;x++){
//                                         //                      tempsum=0;

//                                         //                      for(int i=0;i<MAXFPS_LIM;i++)
//                                         //                                      tempsum+=((int)(tempframesCPUV2[i].at<uchar>(y,x))) ;

//                                         //                      tempsum/=MAXFPS_LIM ;
//                                         //                      sum_array[y][x] = (char)tempsum ;
//                                         //              }
//                                         // }

//                                         // Mat A(height,width,CV_8UC1,sum_array);
//                                         // std::memcpy(A.data,sum_array,height*width*sizeof(char));
//                                         // //imwrite("Avgphoto.jpeg",A) ;
//                                         // Mat Aframe ;
//                                         // A.convertTo(Aframe,CV_64FC1) ;
//                                         // Mat Bframe=Aframe.clone();

//                                         // sum_tempframes-=tempframes_total[loop_counter%DYNAMIC_TIME]/DYNAMIC_TIME;
//                                         // tempframes_total[loop_counter%DYNAMIC_TIME]=Bframe;
//                                         // sum_tempframes+=Bframe/DYNAMIC_TIME;
//                                         // //imwrite("sumframes.jpeg",sum_tempframes);
//                                         // sum_tempframes.convertTo(dynAVG,CV_8UC1);
//                                         //imwrite("avg.jpeg",dynAVG) ;

//                                         ///DYNAVG CODE OVER.... dynAVG is not correct
//                                         // cout<<"starting this 1"<<endl;

//                                         Mat temp1 = cv::abs(blackwhiteframe-staticAVG);
//                                         // cout<<"starting this 2"<<endl;

//                                         // cout<<"blackwhiteframe.size(): "<<blackwhiteframe.size()<<" and dynAVG.size(): "<<dynAVG.size()<<"dynAVG.channels(): "<<dynAVG.channels()<<"blackwhiteframe.channels(): "<<blackwhiteframe.channels()<<endl;

//                                         // cout<<"blackwhiteframe.type(): "<<type2str(blackwhiteframe.type())<<" and dynAVG.type(): "<<type2str(dynAVG.type())<<endl;
//                                         dynAVG.convertTo(dynAVG, CV_8UC1);
//                                         // Mat temp2 = cv::abs(blackwhiteframe-dynAVG);

//                                         t2 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                         t = chrono::high_resolution_clock::now();


//                                         // cout<<"starting this 3"<<endl;
//                                         threshold(temp1,staticAVG_dif,VARIANCE_FROM_AVG,255,THRESH_BINARY) ;
//                                         // cout<<"starting this 4"<<endl;
//                                         // threshold(dynAVG,dynAVG_dif,VARIANCE_FROM_AVG,255,THRESH_BINARY) ;
//                                         // cout<<"starting this 5"<<endl;

//                                         t3 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                         t = chrono::high_resolution_clock::now();

//                                         //imwrite("dynavgdiff.jpeg",dynAVG_dif);

//                                         // bool isRED = isRed(proj(staticAVG_dif,false,SS),proj(dynAVG_dif,false,SS));
//                                         if(SS_ON){
//                                                 projDYN = proj(dynAVG,0,SS);
//                                                 projSTAT = proj(staticAVG_dif,0,SS) ;
//                                         }else{
//                                                 projDYN = proj(dynAVG,0);
//                                                 projSTAT = proj(staticAVG_dif,0) ;
//                                         }

//                                         t4 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                         t = chrono::high_resolution_clock::now();

//                                         isRED = isRed_(projSTAT,projDYN);
//                                         white_area_variable = white_area(projSTAT);
//                                         white_area_variable1 = white_area_dyn(projDYN);
//                                         white_area_variable1 = (white_area_variable1>white_area_variable)?white_area_variable:white_area_variable1;
//                                         stop_density = white_area_variable - white_area_variable1;

//                                         find_line(projSTAT) ;
//                                         line(projSTAT,Point(0,Y),Point(projSTAT.cols,Y),Scalar(255),1) ;
//                                         sy=Y ;
//                                         find_line(projDYN) ;
//                                         dy=Y ;

//                                         t5 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                         t = chrono::high_resolution_clock::now();


//                                         //imshow("Original",blackwhiteframe) ;
//                                         //imshow("Average",dynAVG) ;
//                                         //imshow("Difframe",dynAVG_dif) ;

//                                         //****************************************   U N C O M M E N T   T H E   F O L L O W I N G   F O R    D E  B U G G I N G   *********************************************
//                                 //      imshow("Original_Proj",proj(frame,1)) ;

//                                 //      Mat stop_dens = projSTAT - projDYN;
//                                 //      stop_dens.setTo(0,stop_dens<0);
// //                                      imshow("STOP DENSITY", stop_dens);
// //
// //                                      imshow("DYNAMIC",projDYN) ;
// //
// //                                      imshow("STATIC",projSTAT) ;  // ANSWER FRAME
// //                                      waitKey(0);


//                                         // cout<<"Static_Y: "<<sy<<","<<"Dynamic_Y: "<<dy<<endl ;

//                                         //writing part
//                                         string output_ans = to_string(loop)+":"+to_string(white_area_variable)+","+to_string(white_area_variable1)+","+to_string(sy*1.0/projSTAT.rows) ;
//                                         outfile<<output_ans<<endl ;
//                                         cout<<output_ans<<endl;
//                                 }
//                                         cout<<loop++<<endl;

//                                 // while(WAIT_ON){
//                         //      while(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-startloop).count()/(loop_counter*1.0) < 1000000){};

//                                 // }

//                                 // t6 += (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-t).count());
//                                 // t = chrono::high_resolution_clock::now();

//                                 cout<<t1<<" "<<t2<<endl;//" "<<t3<<" "<<t4<<" "<<t5<<" "<<t6<<" "<<t7<<" "<<t8<<" "<<t9<<endl;
//                            double timetaken= chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-startloop).count();
//                                 cout<<"The final time taken is "<<timetaken/1000000<<endl;
//                                 // if(waitKey(1)==27) break ;
//         //                      cout<<"total time taken: "<<fixed<<setprecision(4)<<(double)
// //(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-startloop).count())<<endl;
//                                 loop_counter++;

//                                 if(loop_counter==5){
//                                         // waitKey(0);
//                                         time1   = chrono::high_resolution_clock::now();

//                                 }
//                                 else if(loop_counter==105){
//                                         cout<<"time for 100 loops: "<<(double)(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-time1).count())<<endl;
//                                         // break;
//                                 }
//                         }

//                         //imwrite("Size_det.jpg",frame) ;
//                         cout<<"I am out of the loop,Press Esc to exit!"<<endl ;
//                         cout<<"Total time to run code:"<<chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now()-startcode).count()<<endl;

//                         if(waitKey(1000)==27) {
//                                 infile.close();
//                                 outfile.close();
//                                 return 0 ;
//                         }
            }

        return 0;
}

