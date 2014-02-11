#pragma once

#include <vector>
#include <memory>
#include <string>

#include "feed.h"
#include "tracker.h"
#include "detector.h"

namespace ch {

	class config {
	};

	class system {
	public:
		system(ch::feed * _feed, ch::detector_base * _detector, ch::tracker * _tracker);
		system(const std::string config_file);
		~system();

		bool is_feed_open();
		const cv::Mat& get_feed();
		const std::string& get_feed_name();
		void update_feed();

		const std::vector<cv::Point2f>& tracker_predict();
		const std::vector<std::pair<std::size_t,cv::Point2f>> tracker_correct(const std::vector<ch::bboxes>& detections);
		
		const std::vector<ch::bboxes> detector_detect(const cv::Mat& image);
		void display_detections(const cv::Mat& image, bool wait_key);


	private:
		ch::feed * feed;
		ch::detector_base * detector; 
		ch::tracker * tracker;
	};
}