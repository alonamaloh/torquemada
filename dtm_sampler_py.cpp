#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dtm_sampler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dtm_sampler, m) {
  m.doc() = "DTM tablebase sampler for training";

  py::class_<DTMSampler>(m, "DTMSampler")
    .def(py::init<>())
    .def("load", &DTMSampler::load,
         py::arg("directory"),
         "Load all DTM tablebase files from directory")
    .def("total_positions", &DTMSampler::total_positions,
         "Total number of positions across all tables")
    .def("num_tables", &DTMSampler::num_tables,
         "Number of loaded tables")
    .def("sample_batch", [](DTMSampler& self, int batch_size) {
      // Create numpy arrays for output
      py::array_t<float> features({batch_size, 128});
      py::array_t<int> classes(batch_size);

      auto feat_buf = features.mutable_unchecked<2>();
      auto cls_buf = classes.mutable_unchecked<1>();

      self.sample_batch(batch_size, feat_buf.mutable_data(0, 0), cls_buf.mutable_data(0));

      return py::make_tuple(features, classes);
    },
    py::arg("batch_size"),
    "Sample a batch of (features, classes) from the tablebases");

  // Expose class constants
  m.attr("NUM_DTM_CLASSES") = NUM_DTM_CLASSES;
  m.attr("DTM_CLASS_DRAW") = DTM_CLASS_DRAW;

  // Class names for display
  m.attr("DTM_CLASS_NAMES") = py::make_tuple(
    "WIN_1", "WIN_2_3", "WIN_4_7", "WIN_8_15", "WIN_16_31", "WIN_32_63", "WIN_64_127",
    "DRAW",
    "LOSS_64_127", "LOSS_32_63", "LOSS_16_31", "LOSS_8_15", "LOSS_4_7", "LOSS_2_3", "LOSS_1"
  );
}
