
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

	const std::vector<std::string> feed::read_directory(const std::string& directory_name) {		
		std::vector<std::string> filenames;

		struct _finddata_t s_file;
		std::string str = directory_name + "\\*.*";
		
		intptr_t h_file = _findfirst( str.c_str(), &s_file );
		if( h_file != static_cast<intptr_t>(-1.0) ) {
			for (unsigned int i = 0; i < 2; ++i) {
				_findnext( h_file, &s_file );
			}
			do {
					filenames.push_back(feed_name+"\\"+(std::string)s_file.name);
			}
		    while( _findnext( h_file, &s_file ) == 0 );
		}
		_findclose( h_file );

		return filenames;
	}

	
}