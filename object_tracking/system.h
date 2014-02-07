#pragma once

#include <vector>
#include <memory>

#include "tracker.h"
#include "detector.h"

namespace ch {

	class system {
	public:
		system(ch::detector_base * &_detector, ch::tracker * &_tracker); 

	private:
		std::unique_ptr<ch::detector_base> detector; 
		std::unique_ptr<ch::tracker> tracker;
	};
}