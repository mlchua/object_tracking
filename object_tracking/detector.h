#pragma once

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

namespace ch {
	
	enum class DET_TYPE {LSVM};

	class bboxes {
	public:
		bboxes();
		bboxes(cv::Rect t_rect, float t_score, int t_id);
		cv::Rect rect;
		float score;
		int classID;
	};

	class detections {
	public:
		detections();
		detections(std::vector<cv::LatentSvmDetector::ObjectDetection> detections);
		std::vector<bboxes> boxes;
	private:
		bboxes lsvmdet_to_bbox(cv::LatentSvmDetector::ObjectDetection lsvmdet);
	};

	class detector_base {
	public:
		virtual ch::detections detect(cv::Mat& image) = 0;
		virtual void display(cv::Mat& image, bool wait_key) = 0;
		DET_TYPE get_det_type() { return type; }
		double get_detect_time() { return detection_time; }
	protected:
		DET_TYPE type;
		double detection_time;
	};

	
	class lsvm_detector : public detector_base {
	public:
		lsvm_detector(float thresh_val = 0.2f);
		lsvm_detector(const std::vector<std::string>& models, float thresh_val = 0.2f);

		ch::detections detect(cv::Mat& image);
		void display(cv::Mat& image, bool wait_key = false);
		bool load_models(const std::vector<std::string>& models);

		const std::vector<std::string>& get_class_names() {return detector.getClassNames();}
		size_t get_class_count() {return detector.getClassCount();}
	private:
		float overlap_threshold;
		cv::LatentSvmDetector detector;
		std::vector<cv::LatentSvmDetector::ObjectDetection> detections;
		std::vector<cv::Scalar> colors;
	};

}