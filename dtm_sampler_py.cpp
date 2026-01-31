#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dtm_sampler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dtm_sampler, m) {
  m.doc() = "DTM tablebase sampler for WDL training";

  py::class_<DTMSampler>(m, "DTMSampler")
    .def(py::init<>())
    .def("load", &DTMSampler::load,
         py::arg("directory"),
         py::arg("min_pieces") = 2,
         py::arg("max_pieces") = 7,
         "Load DTM tablebase files from directory (filtered by piece count)")
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
    "Sample a batch of (features, wdl_classes) from the tablebases");

  // WDL class constants (same as 8+ piece model)
  m.attr("NUM_CLASSES") = NUM_WDL_CLASSES;
  m.attr("WDL_LOSS") = WDL_LOSS;
  m.attr("WDL_DRAW") = WDL_DRAW;
  m.attr("WDL_WIN") = WDL_WIN;

  // Class names for display
  m.attr("CLASS_NAMES") = py::make_tuple("LOSS", "DRAW", "WIN");
}
