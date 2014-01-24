#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "feed.h"
#include "misc.h"
#include "detector.h"
#include <memory>

int main(int argc, char* argv[])
{ 
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

	std::unique_ptr<ch::detector_base> lsvm(new ch::lsvm(models, -1.0f));
	ch::feed feed(images_folder);

	std::cout << "Models loaded: " << std::endl;
	for (auto iter : lsvm->get_class_names()) {
		std::cout << iter << std::endl;
	}

	while (feed.is_open()) {
		std::cout << "Processing: " << feed.get_current_name() << std::endl;
		std::vector<ch::bboxes> detections = lsvm->detect(feed.get_current_image());
		for (auto iter : detections) {
			std::cout << iter.rect.x << '\t' << iter.rect.y << '\t';
			std::cout << iter.classID << '\t' << iter.score << std::endl;
		}
		lsvm->display_detections(feed.get_current_image(), false);
		++feed;
	}
	
	return 0;
}