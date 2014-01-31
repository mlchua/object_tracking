
#pragma once

#include <vector>
#include <string>
#include <iostream>

namespace ch {

	std::vector<std::string> readDirectory(const std::string& directoryName, bool addDirectoryName=true);
	void test_tracker();
}