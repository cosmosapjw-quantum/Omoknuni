#include "batch_evaluator.h"
#include <chrono>
#include <sstream>

namespace alphazero {

// Enable/disable debug logging
constexpr bool DEBUG_LOGGING = false;

// Log a message if debugging is enabled
void debug_log(const std::string& message) {
    if (DEBUG_LOGGING) {
        std::cerr << "[BatchEvaluator] " << message << std::endl;
    }
}

// No additional implementation needed as most functionality is in the header

} // namespace alphazero