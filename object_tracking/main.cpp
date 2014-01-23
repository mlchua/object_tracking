#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "misc.h"
#include "detector.h"

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

	std::vector<std::string> images_filenames = ch::readDirectory(images_folder);
	std::vector<std::string> models_filenames = ch::readDirectory(models_folder);

	ch::lsvm_detector lsvm(models_filenames);
	
	std::cout << "Models: " << std::endl;
	for (auto iter : lsvm.get_class_names()) {
		std::cout << iter << std::endl;
	}
	
	for (auto iter : images_filenames) {
		cv::Mat image = cv::imread(iter);	
		if (image.empty()) { 
			continue;
		}

		std::cout << "Processing: " << iter << std::endl;
		ch::detections detections = lsvm.detect(image);
		lsvm.display(image, true);

	}
	
	return 0;
}