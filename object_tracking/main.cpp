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
#include "system.h"


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

	ch::system system(new ch::feed(images_folder),
		new ch::lsvm(models, detect_th, overlap_th),
		new ch::tracker);

	// Start iterating through our feed
	while (system.is_feed_open()) {
		std::cout << "Processing: " << system.get_feed_name() << std::endl;

		std::vector<cv::Point2f> predictions = system.tracker_predict();
		std::vector<ch::bboxes> detections = system.detector_detect(system.get_feed());
		std::vector<std::pair<std::size_t,cv::Point2f>> corrections = 
			system.tracker_correct(detections);

		std::size_t index = 0;
		std::cout << "Det#\tScore" << std::endl;
		for (auto iter : detections) {
			std::cout << "Det" << index << "\t";
			std::cout << iter.score << std::endl;
			++index;
		}

		system.display_detections(system.get_feed(), false);
		system.update_feed();
	}
	
	return 0;
}