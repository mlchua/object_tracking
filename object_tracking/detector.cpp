#include "detector.h"
#include <string>
#include <iostream>

namespace ch {

	void lsvm_detector::detect(cv::Mat& image) {
		cv::TickMeter meter;
		meter.start();
		detector.detect(image, detections, overlap_threshold);
		meter.stop();
		detection_time = meter.getTimeSec();
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