#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "feed.h"
#include "misc.h"
#include "detector.h"
#include "tracker.h"


int main(int argc, char* argv[])
{ 
	// Read input parameters
	std::string images_folder, models_folder;
	if( argc > 2 ) {
		images_folder = argv[1];
		models_folder = argv[2];
	}

	else {
		std::cout << std::endl << "Using default images and models folders." << std::endl;
		images_folder = "images";
		models_folder = "models";
	}

	std::vector<std::string> models = ch::readDirectory(models_folder);

	// Create our detector and tracker
	const float detect_th =  0.0f;
	const float overlap_th = 0.5f;
	std::unique_ptr<ch::detector_base> lsvm(new ch::lsvm(models, detect_th, overlap_th));
	std::unique_ptr<ch::tracker> kalman(new ch::tracker);
	ch::feed feed(images_folder);

	std::cout << "Models loaded: " << std::endl;
	for (auto iter : lsvm->get_class_names()) {
		std::cout << iter << std::endl;
	}

	// Start iterating through our feed
	while (feed.is_open()) {
		std::cout << "Processing: " << feed.get_current_name() << std::endl;

		std::vector<cv::Point2f> predictions = kalman->predict();
		std::vector<ch::bboxes> detections = lsvm->detect(feed.get_current_image());
		std::vector<std::pair<std::size_t,cv::Point2f>> corrections = 
		kalman->correct(detections);

		std::size_t index = 0;
		std::cout << "Det#\tScore" << std::endl;
		for (auto iter : detections) {
			std::cout << "Det" << index << "\t";
			std::cout << iter.score << std::endl;
			++index;
		}

		lsvm->display_detections(feed.get_current_image(), false);
		++feed;
	}
	
	return 0;
}