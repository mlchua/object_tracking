#pragma once

#include "opencv2\video\tracking.hpp"
#include "opencv2\highgui\highgui.hpp"
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
		void correct(const std::vector<ch::bboxes>& detections);
		std::vector<float> find_lms_scores(const cv::Point2f& track_xy, const std::vector<cv::Point2f>& dets_xy);
		void add_trackers(const std::size_t count);
		std::vector<cv::KalmanFilter> trackers;
	};
}