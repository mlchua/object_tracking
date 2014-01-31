#include "misc.h"

#include <algorithm>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "tracker.h"

#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
#endif

namespace ch {

	std::vector<std::string> readDirectory(const std::string& directoryName, bool addDirectoryName) {
		std::vector<std::string> filenames;

		#if defined(WIN32) | defined(_WIN32)
		    struct _finddata_t s_file;
		    std::string str = directoryName + "\\*.*";
		
		    intptr_t h_file = _findfirst( str.c_str(), &s_file );
		    if( h_file != static_cast<intptr_t>(-1.0) )
		    {
		        do
		        {
		            if( addDirectoryName )
		                filenames.push_back(directoryName + "\\" + s_file.name);
		            else
		                filenames.push_back((std::string)s_file.name);
		        }
		        while( _findnext( h_file, &s_file ) == 0 );
		    }
		    _findclose( h_file );
		#else
		    DIR* dir = opendir( directoryName.c_str() );
		    if( dir != NULL )
		    {
		        struct dirent* dent;
		        while( (dent = readdir(dir)) != NULL )
		        {
		            if( addDirectoryName )
		                filenames.push_back( directoryName + "/" + string(dent->d_name) );
		            else
		                filenames.push_back( string(dent->d_name) );
		        }
		
		        closedir( dir );
		    }
		#endif
		
		    std::sort( filenames.begin(), filenames.end() );
		
			return filenames;
			}

	void test_tracker() {
		ch::tracker track;
		const int lw = 800;
		const int step = 10;
		cv::Mat img(lw,lw, CV_8UC3, cv::Scalar(0,0,0));

		std::vector<std::pair<int,int>> first;
		for (int x = 0; x < lw; x += step) {
			int y = x;
			int err_x = (rand() % 8) - 4;
			int err_y = (rand() % 8) - 4;
			int mea_x = x;
			int mea_y = y;
			if (x+err_x > 0 && x+err_x < lw) { mea_x += err_x; }
			if (y+err_y > 0 && y+err_y < lw) { mea_y += err_y; }
			std::pair<int,int> _t(mea_x,mea_y);
			first.push_back(_t);
		}

		std::vector<std::pair<int,int>> second;
		for (int x = 0; x < lw; x += step) {
			int y = lw - x - 1;
			int err_x = (rand() % 8) - 4;
			int err_y = (rand() % 8) - 4;
			int mea_x = x;
			int mea_y = y;
			if (x+err_x > 0 && x+err_x < lw) { mea_x += err_x; }
			if (y+err_y > 0 && y+err_y < lw) { mea_y += err_y; }
			std::pair<int,int> _t(mea_x,mea_y);
			second.push_back(_t);
		}

		std::vector<std::pair<int,int>> third;
		for (int x = 0; x < lw; x += step) {
			int y = 250;
			int err_x = (rand() % 8) - 4;
			int err_y = (rand() % 8) - 4;
			int mea_x = x;
			int mea_y = y;
			if (x+err_x > 0 && x+err_x < lw) { mea_x += err_x; }
			if (y+err_y > 0 && y+err_y < lw) { mea_y += err_y; }
			std::pair<int,int> _t(mea_x,mea_y);
			third.push_back(_t);
		}

		for (std::size_t t = 0; t < first.size(); ++t) {
			cv::circle( img, cv::Point(first[t].second, first[t].first), 3, cv::Scalar(0,0,255), -1);
			std::cout << "Red:\tX:\t" << first[t].first << "\tY:\t" << first[t].second << std::endl;

			cv::circle( img, cv::Point(second[t].second,second[t].first), 3, cv::Scalar(0,255,0), -1);
			std::cout << "Green:\tX:\t" << second[t].first << "\tY:\t" << second[t].second << std::endl;

			cv::circle( img, cv::Point(third[t].second,third[t].first), 3, cv::Scalar(255,0,0), -1);
			std::cout << "Blue:\tX:\t" << third[t].first << "\tY:\t" << third[t].second << std::endl;

			track.predict();

			std::vector<ch::bboxes> detections;
			detections.push_back(ch::bboxes(cv::Rect(first[t].first,first[t].second,0,0), 1.0f, 0));
			detections.push_back(ch::bboxes(cv::Rect(second[t].first,second[t].second,0,0), 1.0f, 0));
			detections.push_back(ch::bboxes(cv::Rect(third[t].first,third[t].second,0,0), 1.0f, 0));

			std::vector<std::pair<std::size_t,cv::Point2f>> corrects = track.correct(detections);

			for (auto it : corrects) {
				int c_x = it.second.x;
				int c_y = it.second.y;
				c_x = (c_x < 0) ? 0 : c_x;
				c_x = (c_x > lw) ? lw-1 : c_x;
				c_y = (c_y < 0) ? 0 : c_y;
				c_y = (c_y > lw) ? lw-1 : c_y;
				if (it.first == 0) {
					cv::circle( img, cv::Point(c_y, c_x), 2, cv::Scalar(0,0,255/3), -1);
					std::cout << "Tr: 0\tX:\t" << c_x << "\tY:\t" << c_y << std::endl;
				}
				else if (it.first == 1) {
					cv::circle( img, cv::Point(c_y, c_x), 2, cv::Scalar(255/3,0,0), -1);
					std::cout << "Tr: 1\tX:\t" << c_x << "\tY:\t" << c_y << std::endl;
				}
				else {
					cv::circle( img, cv::Point(c_y, c_x), 2, cv::Scalar(0,255/3,0), -1);
					std::cout << "Tr: 2\tX:\t" << c_x << "\tY:\t" << c_y << std::endl;
				}
			}
			
			imshow("Image", img);
			std::string out = "results/track" + std::to_string(t) + ".png";
			cv::imwrite(out, img);
			cvWaitKey(300);
			std::cout << std::endl << std::endl << std::endl;
		}
	}

}