#include "system.h"

#include <fstream>

namespace ch {

	system::system(ch::feed * _feed, ch::detector_base * _detector, ch::tracker * _tracker) :
		feed(_feed),
		detector(_detector),
		tracker(_tracker)
	{};

	system::system(const std::string config_file) {
		std::ifstream config(config_file, std::ios::in);
	}

	system::~system() {
		if (feed != nullptr) {
			delete feed;
		}
		if (detector != nullptr) {
			delete detector;
		}
		if (tracker != nullptr) {
			delete tracker;
		}
	}

	bool system::is_feed_open() {
		return feed->is_open();
	}

	const cv::Mat& system::get_feed() {
		return feed->get_current_image();
	}

	const std::string& system::get_feed_name() {
		return feed->get_current_name();
	}

	void system::update_feed() {
		++(*feed);
	}

	const std::vector<cv::Point2f>& system::tracker_predict() {
		return tracker->predict();
	}

	const std::vector<std::pair<std::size_t,cv::Point2f>> system::tracker_correct(const std::vector<ch::bboxes>& detections) {
		return tracker->correct(detections);
	}

	const std::vector<ch::bboxes> system::detector_detect(const cv::Mat& image) {
		return detector->detect(image);
	}

	void system::display_detections(const cv::Mat& image, bool wait_key = false) {
		detector->display_detections(image, wait_key);
	}

}