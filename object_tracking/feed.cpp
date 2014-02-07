
#include "feed.h"
#include "misc.h"

#include <algorithm>

#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
#endif

namespace ch {

	feed::feed(const std::string _feed_name) : feed_name(_feed_name) {
		feed_list = ch::readDirectory(feed_name); 
		feed_list.erase(feed_list.begin());
		feed_list.erase(feed_list.begin());
		iterator = feed_list.begin();
		image=cv::imread(*iterator);
	}

	const cv::Mat& feed::get_current_image() {
		return image;
	}

	const std::string& feed::get_current_name() {
		return *iterator;
	}

	const bool feed::is_open() {
		return (iterator != feed_list.end() ? true : false);
	}

	feed& feed::operator++() {
		++iterator;
		if (iterator != feed_list.end()) {
			image = cv::imread(*iterator);
		}
		return *this;
	}

}