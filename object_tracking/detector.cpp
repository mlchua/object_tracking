#include "detector.h"
#include <string>
#include <iostream>
#include <vector>

namespace ch {

	bboxes::bboxes(cv::Rect t_rect, float t_score, int t_id) {
		rect = t_rect;
		score = t_score;
		classID = t_id;
	}

	bboxes detections::lsvmdet_to_bbox(cv::LatentSvmDetector::ObjectDetection lsvmdet) {
		bboxes new_box(lsvmdet.rect, lsvmdet.score, lsvmdet.classID);
		return new_box;
	}

	detections::detections(std::vector<cv::LatentSvmDetector::ObjectDetection> detections) {
		for (auto r_iter = detections.rbegin(); r_iter != detections.rend(); ++r_iter) {
			boxes.push_back(lsvmdet_to_bbox(*r_iter));
		}
	}

	lsvm_detector::lsvm_detector(float thresh_val) {
		type = DET_TYPE::LSVM;
		overlap_threshold = thresh_val;
	}

	lsvm_detector::lsvm_detector(const std::vector<std::string>& models, float thresh_val) { 
		type = DET_TYPE::LSVM; 
		overlap_threshold = thresh_val;
		load_models(models);
	}

	ch::detections lsvm_detector::detect(cv::Mat& image) {
		cv::TickMeter meter;
		meter.start();
		detector.detect(image, detections, overlap_threshold);
		meter.stop();
		detection_time = meter.getTimeSec();

		ch::detections detects(detections);
		return detects;
	}

	void lsvm_detector::display(cv::Mat& image, bool wait_key) {
		cv::generateColors(colors, get_class_count());
		for (auto iter : detections) {
			cv::rectangle(image, iter.rect, colors[iter.classID], 3);
			cv::putText(image, get_class_names()[iter.classID], cv::Point(iter.rect.x+4,iter.rect.y+13), 
				cv::FONT_HERSHEY_SIMPLEX, 0.35, colors[iter.classID], 1);
		}
		cv::imshow("result", image);
		if (wait_key) {
			cv::waitKey(0);
		}
	}

	bool lsvm_detector::load_models(const std::vector<std::string>& models) {
		if (!models.empty() && detector.load(models) == true) {
			return true;
		}
		else {
			return false;
		}
	}

}