#include "system.h"

namespace ch {

	system::system(ch::detector_base * &_detector, ch::tracker * &_tracker) :
		detector(std::move(_detector)),
		tracker(std::move(tracker))	
	{};

}