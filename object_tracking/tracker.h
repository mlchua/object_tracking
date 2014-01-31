#pragma once

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "detector.h"
#include <vector>
#include <utility>

namespace ch {

	class tracker {
	public:
		void update(const std::vector<ch::bboxes>& detections);

	private:
		std::vector<cv::Point2f> predict();
		std::vector<int> assign_detections(const std::vector<cv::Point2f>& pres, const std::vector<ch::bboxes>& dets);
		std::vector<std::vector<float>> compute_lms_net(const std::vector<cv::Point2f>& pres, const std::vector<ch::bboxes>& dets);
		std::vector<std::size_t> tracker::sort_index_by_min(const std::vector<float>& scores);
		
		void add_trackers(const std::size_t count);

		std::vector<cv::KalmanFilter> trackers;
	};
}