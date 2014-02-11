#pragma once

#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

namespace ch {

	class feed {
	public:
		feed(const std::string _feed_name);
		const cv::Mat& get_current_image();
		const std::string& get_current_name();
		bool is_open();
		feed& operator++();
	private:

		cv::Mat image;
		std::string feed_name;
		std::vector<std::string> feed_list;
		std::vector<std::string>::iterator iterator;
	};
}