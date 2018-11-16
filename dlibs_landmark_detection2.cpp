
// https://sourceforge.net/p/dclib/discussion/442518/thread/9ad6f091/?limit=25

// 20160930. 68개 landmarks + face rect + dat파일 주로소 호출.
#include "stdint.h"
#include "matrix.h"
#include "mex.h"
#include "opencv2/core/core.hpp"
//#include <opencv2\highgui\highgui.hpp>
//#include "C:/opencv/build/include/opencv2/core/core.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include <algorithm>
#include <iostream>

using namespace dlib;
using namespace std;


const char shape_model[] = "shape_predictor_68_face_landmarks.dat";

bool compare_rect(const rectangle& lhs, const rectangle& rhs)
{
	return lhs.area() < rhs.area();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 1 || nlhs > 1)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Wrong number of arguments");
	if (!mxIsUint8(prhs[0]) || mxIsComplex(prhs[0]))
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Data must be real of type uint8");
	const mwSize *dims = mxGetDimensions(prhs[0]);
	//if (mxGetNumberOfDimensions(prhs[0]) != 3 || dims[2] != 3)
	//	mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "RGB image required");
	if (mxGetNumberOfDimensions(prhs[0]) != 2)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Gray image required");
	if ( mxIsChar(prhs[1]) != 1)
      mexErrMsgTxt("Input 2 must be a string.");

	//mexPrintf("mxGetNumberOfElements(prhs[0]):%d \n",mxGetNumberOfElements(prhs[0]));
	//mexPrintf("mxGetNumberOfDimensions(prhs[0]):%d \n",mxGetNumberOfDimensions(prhs[0]));
	

	//cv::Mat image;
	//image = cv::Mat(dims[1],dims[0],CV_8UC1);
	//const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(prhs[0]));
	//memcpy(image.data, mat_mem_ptr, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	//image = image.t();

	// get the length of the input string //
	//int buflen;
    //buflen = (mxGetM(prhs[1]) * mxGetN(prhs[1])) + 1;
    // copy the string data from input[0] into a C string input_ buf.    //
	char *input_buf;
    input_buf = mxArrayToString(prhs[1]);
    if(input_buf == NULL) 
      mexErrMsgTxt("Could not read HarrCascade Filename to string.");

	const uint8_t *data = reinterpret_cast<const uint8_t*>(mxGetData(prhs[0]));
	// don't know how to construct dlib::array2d directly:(
	//cv::Mat m(dims[0], dims[1], CV_8UC3, (void*)data);
	//cv::Mat m(dims[0], dims[1], CV_8UC1, (void*)data);
	//cv::Mat m(dims[1], dims[0], CV_8UC3, (void*)data);
	//memcpy( m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	cv::Mat m;
	m = cv::Mat(dims[1],dims[0],CV_8UC1);
	memcpy(m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	m = m.t();
	dlib::cv_image<unsigned char> cv_img(m);

	shape_predictor sp;
	//deserialize(shape_model) >> sp;
	deserialize(input_buf) >> sp;

	array2d<unsigned char> img;
	//assign_image(img, dlib::cv_image<bgr_pixel>(m));
	//assign_image(img, dlib::cv_image<unsigned char>(m));
	assign_image(img, cv_img);
	//cv::Mat img1 = cv::imread("Ana_Isabel_Sanchez_0001_gray.jpg");
	//dlib::cv_image<unsigned char> img2(img1);
	//assign_image(img, img2);
	//load_image(img, "Ana_Isabel_Sanchez_0001_gray.jpg");
	//cv::Mat img22 = dlib::toMat(img);
	//for(int i=0; i<250; i++)
	//{
		//for(int j=0; j<250; j++)
		//{
			//mexPrintf("%d ",img22.at<unsigned char>(i,j));
		//}
		//mexPrintf("\n");
	//}
	


	frontal_face_detector detector = get_frontal_face_detector();
	pyramid_up(img);
	std::vector<rectangle> dets = detector(img);

	unsigned long nspartsRect = 68 + 2;
	if (nlhs >= 1)
		plhs[0] = mxCreateDoubleMatrix(dets.size() * nspartsRect, 2, mxREAL);
	
	//mexPrintf("m.size:%d, %d \n",m.size[0],m.size[1]);
	mexPrintf("# of faces:%d \n",dets.size());
	//mexPrintf("nrhs:%d \n",nrhs);
	//mexPrintf("nlhs:%d \n",nlhs);

	if (dets.size() == 0)
	{
		mexPrintf("no face detected\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}

	std::vector<full_object_detection> shapes;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		if (shape.num_parts() == 0)
		{
			mexPrintf("align error\n");
			plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
			//return;
		}
		//for (unsigned long k = 0; k < shape.num_parts(); ++k)
			//mexPrintf("%d, %d \n", shape.part(k).x(), shape.part(k).y() );
			//cout << shape.part(k).x() << " " << shape.part(k).y() << endl;
			//fprintf(outfile, "%d %d\n", shape.part(k).x(), shape.part(k).y() );
		mexPrintf("\n");
		//if (nlhs >= 1)
		//	plhs[0] = mxCreateDoubleMatrix(shape.num_parts()+2, 2, mxREAL);
		double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
		ps[nspartsRect*j + 0] = dets[j].left();
		ps[nspartsRect*j + 1] = dets[j].top();
		ps[nspartsRect*dets.size() + nspartsRect*j + 0] = dets[j].width();
		ps[nspartsRect*dets.size() + nspartsRect*j + 1] = dets[j].height();
		for (int i = 0; i < shape.num_parts(); ++i)
		{
			//ps[2*i] = shape.part(i).x();
			//ps[2*i+1] = shape.part(i).y();
			ps[nspartsRect*j + i+2] = shape.part(i).x();
			ps[nspartsRect*dets.size() + nspartsRect*j + i+2] = shape.part(i).y();
			//mexPrintf("%d, %d\n", shape.part(i).x(), shape.part(i).y() );
		}
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		shapes.push_back(shape);
		//if (nlhs >= 2)
		//	plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);

		//mxGetPr(plhs[1])[0] = dets[j].left();
		//mxGetPr(plhs[1])[1] = dets[j].top();
		//mxGetPr(plhs[1])[2] = dets[j].width();
		//mxGetPr(plhs[1])[3] = dets[j].height();
		////plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);
		////plhs[2] = mxCreateDoubleMatrix(4, 1, mxREAL);
		//double *ps1 = mxGetPr(plhs[1]);
		////double *ps1 = reinterpret_cast<double*>(mxGetData(plhs[1]));
		//ps1[0] = dets[j].left();
		//ps1[1] = dets[j].top();
		//ps1[2] = dets[j].width();
		//ps1[3] = dets[j].height();
	}
	

	
	//const rectangle mCand = *std::max_element(dets.begin(), dets.end(), compare_rect);
	//full_object_detection shape = sp(img, mCand);
	//if (shape.num_parts() == 0)
	//{
	//	mexPrintf("align error\n");
	//	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	//	return;
	//}
	//plhs[0] = mxCreateDoubleMatrix(68, 2, mxREAL);
	//double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
	//for (int i = 0; i < 68; ++i)
	//{
	//	//ps[2*i] = shape.part(i).x();
	//	//ps[2*i+1] = shape.part(i).y();
	//	ps[2*i] = shape.part(i).x();
	//	ps[2*i+1] = shape.part(i).y();
	//	mexPrintf("%d, %d \n", shape.part(i).x(), shape.part(i).y() );
	//}
}
//*/

/*
// 20160929_1. 68개 landmarks + face rect + rgb이미지. but failed
#include "stdint.h"
#include "matrix.h"
#include "mex.h"
#include "opencv2/core/core.hpp"
//#include <opencv2\highgui\highgui.hpp>
//#include "C:/opencv/build/include/opencv2/core/core.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include <algorithm>
#include <iostream>
//#include "opencv_matlab.hpp"

// Matlab-like column-major indexing of 3-D array (be aware of the dimensions: 0<=i<ncols (row) and 0<=j<nrows (column) and - hypothetically - 0<=c<nchannels)
#define _A3D_IDX_COLUMN_MAJOR(i,j,k,nrows,ncols) ((i)+((j)+(k)*ncols)*nrows)
// interleaved row-major indexing for 2-D OpenCV images
//#define _A3D_IDX_OPENCV(x,y,c,mat) (((y)*mat.step[0]) + ((x)*mat.step[1]) + (c))
#define _A3D_IDX_OPENCV(i,j,k,nrows,ncols,nchannels) (((i*ncols + j)*nchannels) + (k))

using namespace dlib;
using namespace std;


const char shape_model[] = "shape_predictor_68_face_landmarks.dat";

bool compare_rect(const rectangle& lhs, const rectangle& rhs)
{
	return lhs.area() < rhs.area();
}

template <typename T>
inline void
	copyMatrixFromMatlab(const T* from, cv::Mat& to)
{
	//assert(to.dims == 2); // =2 <=> 2-D image

	const int dims=to.channels();
	const int rows=to.rows;
	const int cols=to.cols;

	T* pdata = (T*)to.data;

	for (int c = 0; c < dims; c++)
	{
		for (int x = 0; x < cols; x++)
		{
			for (int y = 0; y < rows; y++)
			{
				const T element = from[_A3D_IDX_COLUMN_MAJOR(y,x,c,rows,cols)];
				pdata[_A3D_IDX_OPENCV(y,x,c,rows,cols,dims)] = element;
			}
		}
	}
}

template <typename T>
inline void
	copyMatrixToOpencv(const T* from, cv::Mat& to)
{
	//assert(to.dims == 2); // =2 <=> 2-D image

	copyMatrixFromMatlab(from,to);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 1 || nlhs > 1)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Wrong number of arguments");
	if (!mxIsUint8(prhs[0]) || mxIsComplex(prhs[0]))
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Data must be real of type uint8");
	const mwSize *dims = mxGetDimensions(prhs[0]);
	if (mxGetNumberOfDimensions(prhs[0]) != 3 || dims[2] != 3)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "RGB image required");

	mexPrintf("mxGetNumberOfElements(prhs[0]):%d \n",mxGetNumberOfElements(prhs[0]));

	//cv::Mat image;
	//image = cv::Mat(dims[1],dims[0],CV_8UC1);
	//const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(prhs[0]));
	//memcpy(image.data, mat_mem_ptr, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	//image = image.t();




	const uint8_t *data = reinterpret_cast<const uint8_t*>(mxGetData(prhs[0]));
	// don't know how to construct dlib::array2d directly:(
	//cv::Mat m(dims[0], dims[1], CV_8UC3, (void*)data);
	//cv::Mat m(dims[0], dims[1], CV_8UC1, (void*)data);
	//cv::Mat m(dims[1], dims[0], CV_8UC3, (void*)data);
	//memcpy( m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));

	//cv::Mat m;
	//m = cv::Mat(dims[1],dims[0],CV_8UC3);
	//memcpy(m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	//m = m.t();
	
	//cv::Mat m = cv::Mat(dims[1],dims[0], CV_8UC3, mxGetData(prhs[0]),0);
	//m = m.t();
	cv::Mat m;
    //om::copyMatrixToOpencv(mxGetPr(prhs[0]), m);
    //m.convertTo(m, CV_8U, 255);
	//copyMatrixToOpencv(mxGetPr(prhs[0]), m); 
	copyMatrixToOpencv<uchar>((unsigned char*)mxGetPr(prhs[0]), m);
	m.convertTo(m, CV_8U, 255);


	//cv::Mat m;
	//uint8_t *imgpr = (uint8_t*) mxGetPr(prhs[0]);
	//m = cv::Mat(dims[1],dims[0], CV_8UC3, imgpr).t(); 
	dlib::cv_image<bgr_pixel> cv_img(m);

	shape_predictor sp;
	deserialize(shape_model) >> sp;

	array2d<rgb_pixel> img;
	//assign_image(img, dlib::cv_image<bgr_pixel>(m));
	//assign_image(img, dlib::cv_image<unsigned char>(m));
	assign_image(img, cv_img);
	//cv::Mat img1 = cv::imread("Ana_Isabel_Sanchez_0001_gray.jpg");
	//dlib::cv_image<unsigned char> img2(img1);
	//assign_image(img, img2);
	//load_image(img, "Ana_Isabel_Sanchez_0001_gray.jpg");
	cv::Mat img22 = dlib::toMat(img);
	for(int i=0; i<250; i++)
	{
		for(int j=0; j<250; j++)
		{
			//mexPrintf("%d ",img22.at<unsigned char>(i,j));
			mexPrintf("%d ",m.at<cv::Vec3b>(i,j)[0]);
		}
		mexPrintf("\n");
	}
	


	frontal_face_detector detector = get_frontal_face_detector();
	pyramid_up(img);
	std::vector<rectangle> dets = detector(img);
	
	mexPrintf("m.size:%d, %d \n",m.size[0],m.size[1]);
	mexPrintf("dets.size():%d \n",dets.size());
	mexPrintf("nrhs:%d \n",nrhs);
	mexPrintf("nlhs:%d \n",nlhs);

	if (dets.size() == 0)
	{
		mexPrintf("no face detected\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}

	std::vector<full_object_detection> shapes;
	//for (unsigned long j = 0; j < dets.size(); ++j)
	for (unsigned long j = 0; j < 1; ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		if (shape.num_parts() == 0)
		{
			mexPrintf("align error\n");
			plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
			//return;
		}
		//for (unsigned long k = 0; k < shape.num_parts(); ++k)
			//mexPrintf("%d, %d \n", shape.part(k).x(), shape.part(k).y() );
			//cout << shape.part(k).x() << " " << shape.part(k).y() << endl;
			//fprintf(outfile, "%d %d\n", shape.part(k).x(), shape.part(k).y() );
		mexPrintf("\n");
		if (nlhs >= 1)
			plhs[0] = mxCreateDoubleMatrix(shape.num_parts()+2, 2, mxREAL);
		double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
		ps[0] = dets[j].left();
		ps[1] = dets[j].top();
		ps[shape.num_parts()+2+0] = dets[j].width();
		ps[shape.num_parts()+2+1] = dets[j].height();
		for (int i = 0; i < shape.num_parts(); ++i)
		{
			//ps[2*i] = shape.part(i).x();
			//ps[2*i+1] = shape.part(i).y();
			ps[i+2] = shape.part(i).x();
			ps[shape.num_parts()+i+2+2] = shape.part(i).y();
			mexPrintf("%d, %d\n", shape.part(i).x(), shape.part(i).y() );
		}
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		shapes.push_back(shape);
		//if (nlhs >= 2)
		//	plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);

		//mxGetPr(plhs[1])[0] = dets[j].left();
		//mxGetPr(plhs[1])[1] = dets[j].top();
		//mxGetPr(plhs[1])[2] = dets[j].width();
		//mxGetPr(plhs[1])[3] = dets[j].height();
		////plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);
		////plhs[2] = mxCreateDoubleMatrix(4, 1, mxREAL);
		//double *ps1 = mxGetPr(plhs[1]);
		////double *ps1 = reinterpret_cast<double*>(mxGetData(plhs[1]));
		//ps1[0] = dets[j].left();
		//ps1[1] = dets[j].top();
		//ps1[2] = dets[j].width();
		//ps1[3] = dets[j].height();
	}
}
//*/

/*
// 20160929. 68개 landmarks + face rect.
#include "stdint.h"
#include "matrix.h"
#include "mex.h"
#include "opencv2/core/core.hpp"
//#include <opencv2\highgui\highgui.hpp>
//#include "C:/opencv/build/include/opencv2/core/core.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include <algorithm>
#include <iostream>

using namespace dlib;
using namespace std;


const char shape_model[] = "shape_predictor_68_face_landmarks.dat";

bool compare_rect(const rectangle& lhs, const rectangle& rhs)
{
	return lhs.area() < rhs.area();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 1 || nlhs > 1)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Wrong number of arguments");
	if (!mxIsUint8(prhs[0]) || mxIsComplex(prhs[0]))
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Data must be real of type uint8");
	const mwSize *dims = mxGetDimensions(prhs[0]);
	//if (mxGetNumberOfDimensions(prhs[0]) != 3 || dims[2] != 3)
	//	mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "RGB image required");

	mexPrintf("mxGetNumberOfElements(prhs[0]):%d \n",mxGetNumberOfElements(prhs[0]));

	//cv::Mat image;
	//image = cv::Mat(dims[1],dims[0],CV_8UC1);
	//const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(prhs[0]));
	//memcpy(image.data, mat_mem_ptr, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	//image = image.t();




	const uint8_t *data = reinterpret_cast<const uint8_t*>(mxGetData(prhs[0]));
	// don't know how to construct dlib::array2d directly:(
	//cv::Mat m(dims[0], dims[1], CV_8UC3, (void*)data);
	//cv::Mat m(dims[0], dims[1], CV_8UC1, (void*)data);
	//cv::Mat m(dims[1], dims[0], CV_8UC3, (void*)data);
	//memcpy( m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	cv::Mat m;	m = cv::Mat(dims[1],dims[0],CV_8UC1);
	memcpy(m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	m = m.t();
	dlib::cv_image<unsigned char> cv_img(m);

	shape_predictor sp;
	deserialize(shape_model) >> sp;

	array2d<unsigned char> img;
	//assign_image(img, dlib::cv_image<bgr_pixel>(m));
	//assign_image(img, dlib::cv_image<unsigned char>(m));
	assign_image(img, cv_img);
	//cv::Mat img1 = cv::imread("Ana_Isabel_Sanchez_0001_gray.jpg");
	//dlib::cv_image<unsigned char> img2(img1);
	//assign_image(img, img2);
	//load_image(img, "Ana_Isabel_Sanchez_0001_gray.jpg");
	cv::Mat img22 = dlib::toMat(img);
	for(int i=0; i<250; i++)
	{
		for(int j=0; j<250; j++)
		{
			//mexPrintf("%d ",img22.at<unsigned char>(i,j));
		}
		//mexPrintf("\n");
	}
	


	frontal_face_detector detector = get_frontal_face_detector();
	pyramid_up(img);
	std::vector<rectangle> dets = detector(img);
	
	mexPrintf("m.size:%d, %d \n",m.size[0],m.size[1]);
	mexPrintf("dets.size():%d \n",dets.size());
	mexPrintf("nrhs:%d \n",nrhs);
	mexPrintf("nlhs:%d \n",nlhs);

	if (dets.size() == 0)
	{
		mexPrintf("no face detected\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}

	std::vector<full_object_detection> shapes;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		if (shape.num_parts() == 0)
		{
			mexPrintf("align error\n");
			plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
			//return;
		}
		//for (unsigned long k = 0; k < shape.num_parts(); ++k)
			//mexPrintf("%d, %d \n", shape.part(k).x(), shape.part(k).y() );
			//cout << shape.part(k).x() << " " << shape.part(k).y() << endl;
			//fprintf(outfile, "%d %d\n", shape.part(k).x(), shape.part(k).y() );
		mexPrintf("\n");
		if (nlhs >= 1)
			plhs[0] = mxCreateDoubleMatrix(shape.num_parts()+2, 2, mxREAL);
		double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
		ps[0] = dets[j].left();
		ps[1] = dets[j].top();
		ps[shape.num_parts()+2+0] = dets[j].width();
		ps[shape.num_parts()+2+1] = dets[j].height();
		for (int i = 0; i < shape.num_parts(); ++i)
		{
			//ps[2*i] = shape.part(i).x();
			//ps[2*i+1] = shape.part(i).y();
			ps[i+2] = shape.part(i).x();
			ps[shape.num_parts()+i+2+2] = shape.part(i).y();
			mexPrintf("%d, %d\n", shape.part(i).x(), shape.part(i).y() );
		}
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		shapes.push_back(shape);
		//if (nlhs >= 2)
		//	plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);

		//mxGetPr(plhs[1])[0] = dets[j].left();
		//mxGetPr(plhs[1])[1] = dets[j].top();
		//mxGetPr(plhs[1])[2] = dets[j].width();
		//mxGetPr(plhs[1])[3] = dets[j].height();
		////plhs[1] = mxCreateDoubleMatrix(4, 1, mxREAL);
		////plhs[2] = mxCreateDoubleMatrix(4, 1, mxREAL);
		//double *ps1 = mxGetPr(plhs[1]);
		////double *ps1 = reinterpret_cast<double*>(mxGetData(plhs[1]));
		//ps1[0] = dets[j].left();
		//ps1[1] = dets[j].top();
		//ps1[2] = dets[j].width();
		//ps1[3] = dets[j].height();
	}
	

	
	//const rectangle mCand = *std::max_element(dets.begin(), dets.end(), compare_rect);
	//full_object_detection shape = sp(img, mCand);
	//if (shape.num_parts() == 0)
	//{
	//	mexPrintf("align error\n");
	//	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	//	return;
	//}
	//plhs[0] = mxCreateDoubleMatrix(68, 2, mxREAL);
	//double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
	//for (int i = 0; i < 68; ++i)
	//{
	//	//ps[2*i] = shape.part(i).x();
	//	//ps[2*i+1] = shape.part(i).y();
	//	ps[2*i] = shape.part(i).x();
	//	ps[2*i+1] = shape.part(i).y();
	//	mexPrintf("%d, %d \n", shape.part(i).x(), shape.part(i).y() );
	//}
}
//*/

/*
// 20160928. 68개 landmarks 찾아진 version.
#include "stdint.h"
#include "matrix.h"
#include "mex.h"
#include "opencv2/core/core.hpp"
//#include <opencv2\highgui\highgui.hpp>
//#include "C:/opencv/build/include/opencv2/core/core.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include <algorithm>
#include <iostream>

using namespace dlib;
using namespace std;


const char shape_model[] = "shape_predictor_68_face_landmarks.dat";

bool compare_rect(const rectangle& lhs, const rectangle& rhs)
{
	return lhs.area() < rhs.area();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 1 || nlhs > 1)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Wrong number of arguments");
	if (!mxIsUint8(prhs[0]) || mxIsComplex(prhs[0]))
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Data must be real of type uint8");
	const mwSize *dims = mxGetDimensions(prhs[0]);
	//if (mxGetNumberOfDimensions(prhs[0]) != 3 || dims[2] != 3)
	//	mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "RGB image required");

	mexPrintf("mxGetNumberOfElements(prhs[0]):%d \n",mxGetNumberOfElements(prhs[0]));

	//cv::Mat image;
	//image = cv::Mat(dims[1],dims[0],CV_8UC1);
	//const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(prhs[0]));
	//memcpy(image.data, mat_mem_ptr, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	//image = image.t();




	const uint8_t *data = reinterpret_cast<const uint8_t*>(mxGetData(prhs[0]));
	// don't know how to construct dlib::array2d directly:(
	//cv::Mat m(dims[0], dims[1], CV_8UC3, (void*)data);
	//cv::Mat m(dims[0], dims[1], CV_8UC1, (void*)data);
	//cv::Mat m(dims[1], dims[0], CV_8UC3, (void*)data);
	//memcpy( m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	cv::Mat m;	m = cv::Mat(dims[0],dims[1],CV_8UC1);
	memcpy(m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	m = m.t();
	dlib::cv_image<unsigned char> cv_img(m);

	shape_predictor sp;
	deserialize(shape_model) >> sp;

	array2d<unsigned char> img;
	//assign_image(img, dlib::cv_image<bgr_pixel>(m));
	//assign_image(img, dlib::cv_image<unsigned char>(m));
	assign_image(img, cv_img);
	//cv::Mat img1 = cv::imread("Ana_Isabel_Sanchez_0001_gray.jpg");
	//dlib::cv_image<unsigned char> img2(img1);
	//assign_image(img, img2);
	//load_image(img, "Ana_Isabel_Sanchez_0001_gray.jpg");
	cv::Mat img22 = dlib::toMat(img);
	for(int i=0; i<250; i++)
	{
		for(int j=0; j<250; j++)
		{
			//mexPrintf("%d ",img22.at<unsigned char>(i,j));
		}
		//mexPrintf("\n");
	}
	


	frontal_face_detector detector = get_frontal_face_detector();
	pyramid_up(img);
	std::vector<rectangle> dets = detector(img);
	
	mexPrintf("m.size:%d, %d \n",m.size[0],m.size[1]);
	mexPrintf("dets.size():%d \n",dets.size());
	mexPrintf("dets.crbegin():%d \n",dets.crbegin());

	if (dets.size() == 0)
	{
		mexPrintf("no face detected\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}

	std::vector<full_object_detection> shapes;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		if (shape.num_parts() == 0)
		{
			mexPrintf("align error\n");
			plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
			//return;
		}
		//for (unsigned long k = 0; k < shape.num_parts(); ++k)
			//mexPrintf("%d, %d \n", shape.part(k).x(), shape.part(k).y() );
			//cout << shape.part(k).x() << " " << shape.part(k).y() << endl;
			//fprintf(outfile, "%d %d\n", shape.part(k).x(), shape.part(k).y() );
		mexPrintf("\n");
		plhs[0] = mxCreateDoubleMatrix(shape.num_parts(), 2, mxREAL);
		double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
		for (int i = 0; i < shape.num_parts(); ++i)
		{
			//ps[2*i] = shape.part(i).x();
			//ps[2*i+1] = shape.part(i).y();
			ps[i] = shape.part(i).x();
			ps[shape.num_parts()+i] = shape.part(i).y();
			mexPrintf("%d, %d \n", shape.part(i).x(), shape.part(i).y() );
		}
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		shapes.push_back(shape);
	}

	
	//const rectangle mCand = *std::max_element(dets.begin(), dets.end(), compare_rect);
	//full_object_detection shape = sp(img, mCand);
	//if (shape.num_parts() == 0)
	//{
	//	mexPrintf("align error\n");
	//	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	//	return;
	//}
	//plhs[0] = mxCreateDoubleMatrix(68, 2, mxREAL);
	//double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
	//for (int i = 0; i < 68; ++i)
	//{
	//	//ps[2*i] = shape.part(i).x();
	//	//ps[2*i+1] = shape.part(i).y();
	//	ps[2*i] = shape.part(i).x();
	//	ps[2*i+1] = shape.part(i).y();
	//	mexPrintf("%d, %d \n", shape.part(i).x(), shape.part(i).y() );
	//}
}
//*/


/*
#include "stdint.h"
#include "matrix.h"
#include "mex.h"
#include "opencv2/core/core.hpp"
//#include "C:/opencv/build/include/opencv2/core/core.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include <algorithm>
#include <iostream>

using namespace dlib;
using namespace std;


const char shape_model[] = "shape_predictor_68_face_landmarks.dat";

bool compare_rect(const rectangle& lhs, const rectangle& rhs)
{
	return lhs.area() < rhs.area();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 1 || nlhs > 1)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Wrong number of arguments");
	if (!mxIsUint8(prhs[0]) || mxIsComplex(prhs[0]))
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Data must be real of type uint8");
	const mwSize *dims = mxGetDimensions(prhs[0]);
	if (mxGetNumberOfDimensions(prhs[0]) != 3 || dims[2] != 3)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "RGB image required");

	mexPrintf("mxGetNumberOfElements(prhs[0]):%d \n",mxGetNumberOfElements(prhs[0]));

	

	//cv::Mat image;
	//image = cv::Mat(dims[1],dims[0],CV_8UC1);
	//const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(prhs[0]));
	//memcpy(image.data, mat_mem_ptr, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	//image = image.t();




	const uint8_t *data = reinterpret_cast<uint8_t*>(mxGetData(prhs[0]));
	// don't know how to construct dlib::array2d directly:(
	//cv::Mat m(dims[0], dims[1], CV_8UC3, (void*)data);
	//cv::Mat m(dims[1], dims[0], CV_8UC3, (void*)data);
	//memcpy( m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	cv::Mat m;
	m = cv::Mat(dims[0],dims[1],CV_8UC3);
	memcpy(m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	m = m.t();
	//dlib::cv_image<rgb_pixel> cv_img(m);

	shape_predictor sp;
	deserialize(shape_model) >> sp;

	array2d<rgb_pixel> img;
	assign_image(img, dlib::cv_image<rgb_pixel>(m));
	//assign_image(img, cv_img);

	frontal_face_detector detector = get_frontal_face_detector();
	pyramid_up(img);
	std::vector<rectangle> dets = detector(img);
	
	mexPrintf("m.size:%d, %d \n",m.size[0],m.size[1]);
	mexPrintf("dets.size():%d \n",dets.size());
	mexPrintf("dets.crbegin():%d \n",dets.crbegin());

	if (dets.size() == 0)
	{
		mexPrintf("no face detected\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}
	const rectangle mCand = *std::max_element(dets.begin(), dets.end(), compare_rect);

	full_object_detection shape = sp(img, mCand);
	if (shape.num_parts() == 0)
	{
		mexPrintf("align error\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}
	plhs[0] = mxCreateDoubleMatrix(68, 2, mxREAL);
	double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
	for (int i = 0; i < 68; ++i)
	{
		//ps[2*i] = shape.part(i).x();
		//ps[2*i+1] = shape.part(i).y();
		ps[2*i] = shape.part(i).x();
		ps[2*i+1] = shape.part(i).y();
	}
}
//*/
	/*
	if (nrhs < 1 || nlhs > 1)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Wrong number of arguments");
	if (!mxIsUint8(prhs[0]) || mxIsComplex(prhs[0]))
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "Data must be real of type uint8");
	const mwSize *dims = mxGetDimensions(prhs[0]);
	if (mxGetNumberOfDimensions(prhs[0]) != 3 || dims[2] != 3)
		mexErrMsgIdAndTxt("Dlib:dlib_detect_landmark", "RGB image required");

	mexPrintf("mxGetNumberOfElements(prhs[0]):%d \n",mxGetNumberOfElements(prhs[0]));

	//cv::Mat image;
	//image = cv::Mat(dims[1],dims[0],CV_8UC1);
	//const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(prhs[0]));
	//memcpy(image.data, mat_mem_ptr, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	//image = image.t();




	const uint8_t *data = reinterpret_cast<uint8_t*>(mxGetData(prhs[0]));
	// don't know how to construct dlib::array2d directly:(
	//cv::Mat m(dims[0], dims[1], CV_8UC3, (void*)data);
	//cv::Mat m(dims[1], dims[0], CV_8UC3, (void*)data);
	//memcpy( m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	cv::Mat m;
	m = cv::Mat(dims[0],dims[1],CV_8UC3);
	memcpy(m.data, data, sizeof(unsigned char) * mxGetNumberOfElements(prhs[0]));
	m = m.t();
	//dlib::cv_image<rgb_pixel> cv_img(m);

	shape_predictor sp;
	deserialize(shape_model) >> sp;

	array2d<rgb_pixel> img;
	assign_image(img, dlib::cv_image<rgb_pixel>(m));
	//assign_image(img, cv_img);

	frontal_face_detector detector = get_frontal_face_detector();
	pyramid_up(img);
	std::vector<rectangle> dets = detector(img);
	
	mexPrintf("m.size:%d, %d \n",m.size[0],m.size[1]);
	mexPrintf("dets.size():%d \n",dets.size());

	if (dets.size() == 0)
	{
		mexPrintf("no face detected\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}
	const rectangle mCand = *std::max_element(dets.begin(), dets.end(), compare_rect);

	full_object_detection shape = sp(img, mCand);
	if (shape.num_parts() == 0)
	{
		mexPrintf("align error\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;
	}
	plhs[0] = mxCreateDoubleMatrix(49, 2, mxREAL);
	double *ps = reinterpret_cast<double*>(mxGetData(plhs[0]));
	for (int i = 0; i < 49; ++i)
	{
		ps[2*i] = shape.part(i).x();
		ps[2*i+1] = shape.part(i).y();
	}
	*/