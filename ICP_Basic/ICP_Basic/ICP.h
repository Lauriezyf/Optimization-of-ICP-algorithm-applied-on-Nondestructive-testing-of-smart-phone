#include "cv.h"
#include "highgui.h"
#include "cvaux.h"
#include "cxcore.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void ICP(const vector<Point2f> contour_sample, int num_sample,
	const vector<Point2f> contour_model, int num_model, int col, int row);


void ICP(const vector<Point2f> contour_sample, int num_sample,
	const vector<Point2f> contour_model, int num_model, int col, int row)
{
	time_t ICP_start = time(NULL);

	int iter_time = 1;
	float error = 0;
	float pre_error = FLT_MAX;

	float _R[4], _T[2];

	_R[0] = 1; _R[1] = 0; _T[0] = 0;
	_R[2] = 0; _R[3] = 1; _T[1] = 0;

	//find the nearest point for sample in model for the first time
	vector<Point2f> near_model;

	for (int i = 0; i < num_sample; i++)
	{
		float it1 = 0;

		it1 = (contour_sample[i].x - contour_model[0].x) * (contour_sample[i].x - contour_model[0].x) + (contour_sample[i].y - contour_model[0].y) * (contour_sample[i].y - contour_model[0].y);
		int index = 0;
		for (int j = 1; j < num_model; j++)
		{
			float it2;

			it2 = (contour_sample[i].x - contour_model[j].x) * (contour_sample[i].x - contour_model[j].x) + (contour_sample[i].y - contour_model[j].y) * (contour_sample[i].y - contour_model[j].y);

			if (it2 < it1)
			{
				it1 = it2;
				index = j;
			}
		}

		error += it1;
		near_model.push_back(contour_model[index]);
	}

	printf("No. %d iteration, error = %f\n", iter_time, error);


	vector<Point2f> _sample;
	for (int i = 0; i < num_sample; i++)
	{
		_sample.push_back(contour_sample[i]);
	}


	while ((pre_error - error) > 0)

	{
		pre_error = error;
		error = 0;

		//find the mean of the near_model and sample
		CvPoint2D32f mean_near_model, mean_sample;

		mean_near_model.x = 0;
		mean_near_model.y = 0;
		mean_sample.x = 0;
		mean_sample.y = 0;

		for (int i = 0; i < num_sample; i++)
		{
			mean_near_model.x += near_model[i].x;
			mean_near_model.y += near_model[i].y;
			mean_sample.x += _sample[i].x;
			mean_sample.y += _sample[i].y;
		}

		mean_near_model.x /= (float)num_sample;
		mean_near_model.y /= (float)num_sample;
		mean_sample.x /= (float)num_sample;
		mean_sample.y /= (float)num_sample;

		//recenter the near_model and sample
		float A[4] = { 0, 0, 0, 0 };
		float U[4] = { 0, 0, 0, 0 };
		float W[4] = { 0, 0, 0, 0 };
		float V[4] = { 0, 0, 0, 0 };

		for (int i = 0; i < num_sample; i++)
		{
			float new_near_model_x = near_model[i].x - mean_near_model.x;
			float new_near_model_y = near_model[i].x - mean_near_model.y;
			float new_sample_x = _sample[i].x - mean_sample.x;
			float new_sample_y = _sample[i].x - mean_sample.y;

			A[0] += new_sample_x * new_near_model_x;
			A[1] += new_sample_x * new_near_model_y;
			A[2] += new_sample_y * new_near_model_x;
			A[3] += new_sample_y * new_near_model_y;
		}


		//get transform matrix
		CvMat a_ = cvMat(2, 2, CV_32F, A);
		CvMat u_ = cvMat(2, 2, CV_32F, U);
		CvMat w_ = cvMat(2, 2, CV_32F, W);
		CvMat v_ = cvMat(2, 2, CV_32F, V);

		cvSVD(&a_, &w_, &u_, &v_, CV_SVD_MODIFY_A);

		float R[4], T[2];

		R[0] = V[0] * U[0] + V[1] * U[1];
		R[1] = V[0] * U[2] + V[1] * U[3];
		R[2] = V[2] * U[0] + V[3] * U[1];
		R[3] = V[2] * U[2] + V[3] * U[3];

		T[0] = mean_near_model.x - R[0] * mean_sample.x - R[1] * mean_sample.y;
		T[1] = mean_near_model.y - R[2] * mean_sample.x - R[3] * mean_sample.y;


		//Renew the sample point set    
		for (int i = 0; i < num_sample; i++)
		{
			float pre_x = _sample[i].x;
			float pre_y = _sample[i].y;
			_sample[i].x = R[0] * pre_x + R[1] * pre_y + T[0];
			_sample[i].y = R[2] * pre_x + R[3] * pre_y + T[1];
		}

		//find the nearest point for sample in model
		iter_time++;
		for (int i = 0; i < num_sample; i++)
		{
			float it1 = 0;
			int index = 0;
			it1 = (_sample[i].x - contour_model[0].x) * (_sample[i].x - contour_model[0].x) + (_sample[i].y - contour_model[0].y) * (_sample[i].y - contour_model[0].y);

			for (int j = 1; j < num_model; j++)
			{
				float it2;

				it2 = (_sample[i].x - contour_model[j].x) * (_sample[i].x - contour_model[j].x) + (_sample[i].y - contour_model[j].y) * (_sample[i].y - contour_model[j].y);

				if (it2 < it1)
				{
					it1 = it2;
					index = j;
				}
			}

			near_model[i].x = contour_model[index].x;
			near_model[i].y = contour_model[index].y;
			error += it1;
		}

		printf("No. %d iteration, error = %f\n", iter_time, error);
	}

	time_t ICP_finish = time(NULL);

	cout << "ICP algorithm running time : " << ICP_finish - ICP_start << "seconds" << endl;

	//Store and print the transformed contours
	IplImage *image = cvCreateImage(cvSize(col, row), 8, 3);
	cvSaveImage("image.png", image);

	Mat result = imread("image.png");

	for (int i = 0; i < num_model; i++)
	{
		int mi = (int)contour_model[i].x;
		int mj = (int)contour_model[i].y;
		Vec3b *pixel = result.ptr<Vec3b>(mj);
		pixel[mi][0] = 0;
		pixel[mi][1] = 0;
		pixel[mi][2] = 255;
	}

	for (int i = 0; i < num_sample; i++)
	{
		int si = (int)_sample[i].x;
		int sj = (int)_sample[i].y;
		Vec3b *pixel = result.ptr<Vec3b>(sj);
		pixel[si][0] = 255;
		pixel[si][1] = 0;
		pixel[si][2] = 0;
	}

	imwrite("result.png", result);

	time_t print_finish = time(NULL);

	cout << "Printing time : " << print_finish - ICP_finish << "seconds" << endl;

}