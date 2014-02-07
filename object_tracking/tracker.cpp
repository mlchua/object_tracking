#include "tracker.h"

#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "assert.h"

namespace ch {

	const std::vector<cv::Point2f>& tracker::predict() {
		// Remove past predictions
		if (!predictions.empty()) {
			predictions.clear();
			std::vector<cv::Point2f>(predictions).swap(predictions);
		}	

		// No need to compute if there are no active trackers
		if (trackers.empty()) {
			return predictions;
		}
		
		// Get point predictions for each tracker
		for (auto iter : trackers) {
			cv::Mat pr = iter.predict();
			cv::Point2f pr_xy(pr.at<float>(0), pr.at<float>(1));
			predictions.push_back(pr_xy);
		}

		return predictions;
	}

	std::vector<std::pair<std::size_t,cv::Point2f>> tracker::correct(const std::vector<ch::bboxes>& detections) {

		std::vector<std::pair<std::size_t,cv::Point2f>> corrects;
		// Skip kalman update if no detections
		if (detections.empty()) {
			return corrects;
		}

		// Assign each detection to each tracker
		std::vector<int> claimed = assign_detections(detections);

		// Get corrections
		// TODO: Clean up, temporary code
		for (std::size_t t = 0; t < claimed.size(); ++t) {
			cv::Mat_<float> m(2,1);
			m(0) = static_cast<float>(detections[t].rect.x);
			m(1) = static_cast<float>(detections[t].rect.y);
			cv::Mat estimated = trackers[t].correct(m);
			cv::Point2f _pt(estimated.at<float>(0), estimated.at<float>(1));
			std::pair<std::size_t, cv::Point2f> _t(claimed[t], _pt);
			corrects.push_back(_t);
		}

		return corrects;
	}

	std::vector<int> tracker::assign_detections(const std::vector<ch::bboxes>& detections) {
		// Default value -1 means no detection assigned
		std::vector<int> claimed(detections.size(), -1);

		// Naive, greedy detection assignment
		// TODO: Implement better algorithm, runs O(n^2)
		const int int_max = std::numeric_limits<int>::max();
		const float float_max = std::numeric_limits<float>::max();
		std::vector<ch::bboxes> _dets(detections);
		for (std::size_t i = 0; i < predictions.size(); ++i) {
			float low_score = float_max;
			std::size_t low_idx = 0;
				for (std::size_t j = 0; j < _dets.size(); ++j) {
					// Apply distance formula
					float score = std::sqrt(std::pow(predictions[i].x-_dets[j].rect.x,2)
						+std::pow(predictions[i].y-_dets[j].rect.y,2));
					if (score < low_score) {
						low_score = score;
						low_idx = i;
					}
				}
			claimed[low_idx] = i; 
			_dets[low_idx].rect = cv::Rect(int_max, int_max, 0, 0);
		}

		add_trackers(claimed, detections);

		return claimed;
	}

	std::vector<std::vector<float>> tracker::compute_lms_net(const std::vector<ch::bboxes>& detections) {
		const float fl_max = std::numeric_limits<float>::max();
		// Create our lms score net with default max float val
		std::vector<std::vector<float>> net(predictions.size(), 
			std::vector<float>(detections.size(), fl_max));

		// Compute lms for each (predictions, detections) pair, lower is better
		for (std::size_t i = 0; i < predictions.size(); ++i) {
			for (std::size_t j = 0; j < detections.size(); ++j) {
				float x_sqr = std::pow(predictions[i].x - detections[j].rect.x, 2);
				float y_sqr = std::pow(predictions[i].y - detections[j].rect.y, 2);
				net[i][j] = std::sqrt(x_sqr + y_sqr);
			}
		}

		return net;
	}

	std::vector<std::size_t> tracker::sort_index_by_min(const std::vector<float>& scores) {
		// Create vector with increasing index values
		std::vector<size_t> indices(scores.size());
		std::iota(begin(indices), end(indices), static_cast<size_t>(0));
		
		// Lambda to sort using scores val for comparison
		std::sort( begin(indices), end(indices), 
			[&](size_t a, size_t b) { return scores[a] < scores[b]; } );

		return indices;
	}

	void tracker::add_trackers(std::vector<int>& claimed, const std::vector<ch::bboxes>& detections) {
		// Create a template kalman filter to be inserted
		// TODO: Create config file to load template instead of magic numbers
		cv::KalmanFilter kf(4, 2, 0);

		cv::setIdentity(kf.measurementMatrix);
		cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(100));
		cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(10));
		cv::setIdentity(kf.errorCovPost, cv::Scalar::all(10));

		// Add the new trackers for unclaimed detections
		assert(claimed.size()  == detections.size() && "Claimed and Detections are not same size!");
		for (std::size_t i = 0; i < claimed.size(); ++i) {
			// Check if detection is not yet claimed
			if (claimed[i] == -1) {
				kf.transitionMatrix = 
					*(cv::Mat_<float>(4, 4) << 
					1,0,1,0, 
					0,1,0,1,  
					0,0,1,0,  
					0,0,0,1);
				kf.statePre.at<float>(0) = (float)detections[i].rect.x;
				kf.statePre.at<float>(1) = (float)detections[i].rect.y;
				kf.statePre.at<float>(2) = 1;
				kf.statePre.at<float>(3) = 1;
				trackers.push_back(kf);
				cv::Mat pr = trackers[trackers.size()-1].predict();
				cv::Point2f pr_xy(pr.at<float>(0), pr.at<float>(1));
				claimed[i] = trackers.size() - 1;
			}
		}
	}

}