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
		bboxes(cv::Rect _rect, float _score, int _id);
		cv::Rect rect;
		float score;
		int classID;
	};


	class detector_base {
	public:
		virtual std::vector<ch::bboxes> detect(const cv::Mat& image) = 0;
		virtual void display_detections(const cv::Mat& image, bool wait_key) = 0;

		virtual const std::vector<std::string>& get_class_names() = 0;
		DET_TYPE get_det_type() { return type; }
	protected:
		DET_TYPE type;
	};

	
	class lsvm : public detector_base {
	public:
		lsvm(std::vector<std::string> models, const float _detect_th=-2.0f, float _overlap_th=0.5f);

		std::vector<ch::bboxes> detect(const cv::Mat& image);
		void display_detections(const cv::Mat& image, bool wait_key = false);

		const std::vector<std::string>& get_class_names();
		const std::size_t get_class_count();

		bool load_models(const std::vector<std::string>& models);
	
	private:
		std::size_t count;
		float overlap_th;
		float detect_th;
		cv::LatentSvmDetector detector;
		std::vector<cv::Scalar> colors;
		std::vector<cv::LatentSvmDetector::ObjectDetection> detections;
	};

}