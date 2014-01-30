#include "tracker.h"

#include <cmath>

namespace ch {

	void tracker::update(const std::vector<ch::bboxes>& detections) {
		std::vector<cv::Point2f> predict;
		if (!trackers.empty()) {
			for (auto iter : trackers) {
				cv::Mat p = iter.predict();
				cv::Point2f p_xy(p.at<float>(0), p.at<float>(1));
				predict.push_back(p_xy);
			}
		}

		for (std::size_t i =0; i < trackers.size(); ++i) {
			cv::Point2f tracker_xy(predict[i].x,predict[i].y); 
		}
	}

	void tracker::correct(const std::vector<ch::bboxes>& detections) {
		std::vector<cv::Point2f> dets_xy;

		for(auto iter : detections) {
			cv::Point2f xy((float)iter.rect.x, (float)iter.rect.y);
			dets_xy.push_back(xy);
		}

	}

	void tracker::add_trackers(const std::size_t count) {
		cv::KalmanFilter kf(4, 2, 0);
		kf.transitionMatrix = 
			*(cv::Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1,  0,0,1,0,  0,0,0,1);
		cv::Mat_<float> measurement(2,1);
		measurement.setTo(cv::Scalar(0));

		kf.statePre.at<float>(0) = 0;
		kf.statePre.at<float>(1) = 0;
		kf.statePre.at<float>(2) = 0;
		kf.statePre.at<float>(3) = 0;

		cv::setIdentity(kf.measurementMatrix);
		cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1));
		cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(100));
		cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

		for (std::size_t i = 0; i < count; ++i) {
			trackers.push_back(kf);
		}
	}

	std::vector<float> find_lms_scores(const cv::Point2f& track_xy, const std::vector<cv::Point2f>& dets_xy) {
		std::vector<float> lms_scores(dets_xy.size(), -1.0f);
		
		for (std::size_t i = 0; i < dets_xy.size(); ++i) {
			float x_sqr = std::pow(track_xy.x - dets_xy[i].x, 2);
			float y_sqr = std::pow(track_xy.y - dets_xy[i].y, 2);
			lms_scores[i] = std::sqrt(x_sqr + y_sqr);
		}

		return lms_scores;
	}
}