#include "tracker.h"

#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iostream>

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

		// Temp writing
		std::cout << "Detections" << std::endl;
		for (auto it : detections) {
			std::cout << "X:\t" << it.rect.x << "\tY:\t" << it.rect.y << std::endl;
		}

		std::vector<std::pair<std::size_t,cv::Point2f>> corrects;
		// Skip kalman update if no detections
		if (detections.empty()) {
			return corrects;
		}

		// Add more trackers if more detections than trackers
		if (detections.size() > trackers.size()) {
			add_trackers(detections.size() - trackers.size());
		}

		// Assign each detection to each tracker
		std::vector<int> assignments = assign_detections(detections);

		// Get corrections
		// TODO: Clean up, temporary code
		for (std::size_t t = 0; t < assignments.size(); ++t) {
			if (assignments[t] < 0) {
				break;
			}
			cv::Mat_<float> m(2,1);
			m(0) = detections[assignments[t]].rect.x;
			m(1) = detections[assignments[t]].rect.y;
			cv::Mat estimated = trackers[t].correct(m);
			cv::Point2f _pt(estimated.at<float>(0), estimated.at<float>(1));
			std::pair<std::size_t, cv::Point2f> _t(t, _pt);
			corrects.push_back(_t);
		}

		return corrects;
	}

	std::vector<int> tracker::assign_detections(const std::vector<ch::bboxes>& detections) {
		// Default value -1 means no detection assigned
		std::vector<int> assignments(predictions.size(), -1);

		// Temp code
		std::cout << "Predictions" << std::endl;
		for (auto it : predictions) {
			std::cout << "X:\t" << it.x << "\tY:\t" << it.y << std::endl;
		}

		// Get our lms net for each (predictions, detections) pair
		std::vector<std::vector<float>> net = compute_lms_net(detections);

		// Temp code 
		std::cout << "Net" << std::endl;
		for (auto it : net) {
			for (auto itt : it) {
				std::cout << itt  << '\t';
			}
			std::cout << std::endl;
		}

		// Start assigning detections to predictions
		const float fl_max = std::numeric_limits<float>::max();

		for(std::size_t i = 0; i < predictions.size(); ++i) {
			// Get indices sorted by lowest score to prediction
			std::vector<std::size_t> indices = sort_index_by_min(net[i]);
			// Check if all detections have already been assigned
			if (indices[0] == fl_max) {
				break;
			}
			// Check if score is lowest among all predictions
			// If not, move to next index 
			for(std::size_t j = 0; j < indices.size(); ++j) {
				bool is_lowest = true;
				for(std::size_t k = 0; k < predictions.size(); ++k) {
					if (indices[j] > net[k][indices[j]]) {
						is_lowest = false;
					}
				}
				// If lowest, assign it to prediction and set that det 
				// to max to avoid it being chosen by another prediction
				if (is_lowest) {
					assignments[i] = indices[j];
					for (std::size_t l = 0; l < predictions.size(); ++l) {
						net[l][indices[j]] = fl_max;
					}
					// Move to next prediction
					break;
				}
			}
			// Temp code 
			std::cout << "Net" << std::endl;
			for (auto it : net) {
				for (auto itt : it) {
					std::cout << itt  << '\t';
				}
				std::cout << std::endl;
			}
		}

		// Temp code
		std::cout << "Assignments" << std::endl;
		for (std::size_t i = 0; i < assignments.size(); ++i) {
			std::cout << i << ":\t" << assignments[i] << std::endl;
		}

		return assignments;
	}

	std::vector<std::vector<float>> tracker::compute_lms_net(const std::vector<ch::bboxes>& detections) {
		const float fl_max = std::numeric_limits<float>::max();
		// Create our lms score net with default max float val
		std::vector<std::vector<float>> net(predictions.size(), 
			std::vector<float>(detections.size(), fl_max));

		// Temp code 
		std::cout << "Net" << std::endl;
		for (auto it : net) {
			for (auto itt : it) {
				std::cout << itt  << '\t';
			}
			std::cout << std::endl;
		}

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

	void tracker::add_trackers(const std::size_t count) {
		// Create a template kalman filter to be inserted
		// TODO: Create config file to load template instead of magic numbers
		cv::KalmanFilter kf(4, 2, 0);
		kf.transitionMatrix = 
			*(cv::Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1,  0,0,1,0,  0,0,0,1);

		kf.statePre.at<float>(0) = 0;
		kf.statePre.at<float>(1) = 0;
		kf.statePre.at<float>(2) = 0;
		kf.statePre.at<float>(3) = 0;

		cv::setIdentity(kf.measurementMatrix);
		cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1));
		cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1));
		cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

		// Add the new trackers
		for (std::size_t i = 0; i < count; ++i) {
			trackers.push_back(kf);
		}
	}

}