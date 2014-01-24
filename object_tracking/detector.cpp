#include "detector.h"
#include <string>
#include <iostream>
#include <vector>

namespace ch {

	bboxes::bboxes(cv::Rect _rect, float _score, int _id) 
		: rect(_rect), score(_score), classID(_id) {}

	lsvm::lsvm(std::vector<std::string> models, const float _detect_th, const float _overlap_th) 
		: detect_th(_detect_th), overlap_th(_overlap_th) {
				type=DET_TYPE::LSVM;
				detector.load(models);
				cv::generateColors(colors, get_class_count());
	}

	const std::vector<ch::bboxes> lsvm::detect(const cv::Mat& image) {
		cv::TickMeter meter;
		meter.start();
		detector.detect(image, detections, overlap_th, 4);
		meter.stop();
		detection_time = meter.getTimeSec();

		std::vector<ch::bboxes> bounding_boxes;
		for (auto r_iter = detections.rbegin(); r_iter != detections.rend(); ++r_iter) {
			if (r_iter->score > detect_th) {
				ch::bboxes box(r_iter->rect, r_iter->score, r_iter->classID);
				bounding_boxes.push_back(box);
			}
		}
		return bounding_boxes;
	}

	void lsvm::display_detections(const cv::Mat& image, bool wait_key) {
		const int font_face = cv::FONT_HERSHEY_SIMPLEX;
		const double font_scale = 0.35;

		cv::Mat drawn(image);

		for (auto iter : detections) {
			if (iter.score > detect_th) {
				cv::rectangle(drawn, iter.rect, colors[iter.classID], 3);
				cv::putText(drawn, get_class_names()[iter.classID], 
					cv::Point(iter.rect.x+4,iter.rect.y+13), 
					font_face, font_scale, colors[iter.classID], 1);
			}
		}

		cv::imshow("result", image);
		if (wait_key) {
			cv::waitKey(0);
		}
	}

	bool lsvm::load_models(const std::vector<std::string>& models) {
		return (!models.empty() && detector.load(models) == true) 
			? true : false;
	}

	const std::vector<std::string>& lsvm::get_class_names() {
		return detector.getClassNames();
	}

	const size_t lsvm::get_class_count() {
		return detector.getClassCount();
	}
}