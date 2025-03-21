// Fixed file: alphazero/bindings/batch_evaluator_bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <iostream>
#include "../core/mcts/batch_evaluator.h"

namespace py = pybind11;
namespace alphazero {

// PyTorch batch evaluator implementation
class PyTorchBatchEvaluator : public BatchEvaluator {
private:
    py::object evaluator_;
    
public:
    PyTorchBatchEvaluator(py::object evaluator, int batch_size = 16, int max_wait_ms = 5)
        : BatchEvaluator(batch_size, max_wait_ms), evaluator_(evaluator) {}
    
    std::vector<std::pair<std::vector<float>, float>> evaluate_batch(
        const std::vector<std::vector<float>>& states) override {
        
        try {
            // Call Python batch evaluator
            py::object result = evaluator_.attr("evaluate_batch")(states);
            
            // Extract policies and values - fixed indexing to use py::int_
            py::object policies_obj = result.attr("__getitem__")(py::int_(0));
            py::object values_obj = result.attr("__getitem__")(py::int_(1));
            
            // Make sure we have lists
            py::list policies = policies_obj.cast<py::list>();
            py::list values = values_obj.cast<py::list>();
            
            // Convert to C++ types
            std::vector<std::pair<std::vector<float>, float>> results;
            for (size_t i = 0; i < py::len(policies); ++i) {
                std::vector<float> policy = policies[i].cast<std::vector<float>>();
                float value = values[i].cast<float>();
                results.emplace_back(policy, value);
            }
            
            return results;
        }
        catch (const py::error_already_set& e) {
            std::cerr << "Python error in evaluate_batch: " << e.what() << std::endl;
            return {};
        }
        catch (const std::exception& e) {
            std::cerr << "C++ error in evaluate_batch: " << e.what() << std::endl;
            return {};
        }
    }
};

PYBIND11_MODULE(batch_evaluator, m) {
    py::class_<BatchEvaluator, std::shared_ptr<BatchEvaluator>>(m, "BatchEvaluator")
        .def("submit", &BatchEvaluator::submit);
    
    py::class_<PyTorchBatchEvaluator, BatchEvaluator, std::shared_ptr<PyTorchBatchEvaluator>>(m, "PyTorchBatchEvaluator")
        .def(py::init<py::object, int, int>(),
             py::arg("evaluator"),
             py::arg("batch_size") = 16,
             py::arg("max_wait_ms") = 5);
}

} // namespace alphazero