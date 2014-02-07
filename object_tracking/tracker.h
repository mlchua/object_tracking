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
		const std::vector<cv::Point2f>& predict();
		std::vector<std::pair<std::size_t,cv::Point2f>> correct(const std::vector<ch::bboxes>& detections);

	private:
		std::vector<int> assign_detections(const std::vector<ch::bboxes>& detections);
		std::vector<std::vector<float>> compute_lms_net(const std::vector<ch::bboxes>& detections);
		std::vector<std::size_t> tracker::sort_index_by_min(const std::vector<float>& scores);
		
		void tracker::add_trackers(std::vector<int>& claimed, const std::vector<ch::bboxes>& detections);

		std::vector<cv::KalmanFilter> trackers;
		std::vector<cv::Point2f> predictions;
	};
}