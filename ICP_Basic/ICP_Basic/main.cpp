#include "cv.h"
#include "highgui.h"
#include "cvaux.h"
#include "cxcore.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "ICP.h"


using namespace cv;
using namespace std;



int main()
{
	time_t contour_start = time(NULL);

	//Read the model and sample
	Mat model = imread("model.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat sample = imread("sample.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	//Binary the two picture, using adaptive Threshold
	int blockSize = 25;
	int constValue = 10;

	Mat local_model, local_sample;
	adaptiveThreshold(model, local_model, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
	imwrite("model2.png", local_model);
	adaptiveThreshold(sample, local_sample, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
	imwrite("sample2.png", local_sample);

	//Get contours of the model and the sample
	vector<Point2f> contour_model, contour_sample;
	vector<vector<Point> > contours_model, contours_sample;
	vector<Vec4i> hierarchy;
	findContours(local_model, contours_model, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));
	findContours(local_sample, contours_sample, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));
    
	//storage edge points sequentially
	int numOfContoursformodel = (int)contours_model.size();
	int numOfContoursforSample = (int)contours_sample.size();
	int num_model = 0, num_sample = 0;

	for (int i = 0; i < numOfContoursformodel; i++)
	{
		int len = (int)contours_model[i].size();
		for (int j = 0; j < len; j++)
		{
			contour_model.push_back(contours_model[i][j]);
			num_model++;
		}
	}

	for (int i = 0; i < numOfContoursforSample; i++)
	{
		int len = (int)contours_sample[i].size();
		for (int j = 0; j < len; j++)
		{
			contour_sample.push_back(contours_sample[i][j]);
			num_sample++;
		}
	}

	time_t contour_finish = time(NULL);

	cout << "Reading contours : " << contour_finish - contour_start << "seconds" << endl;
	cout << "Number of model's contours : " << num_model << endl;
	cout << "Number of sample's contours : " << num_sample << endl;

	//Running ICP algorithm
	ICP(contour_model, num_model, contour_sample, num_sample, model.cols, model.rows);

	system("pause");
	return 0;
}

