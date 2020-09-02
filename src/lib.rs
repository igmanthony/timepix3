#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};
use rayon::prelude::*;

mod dbscan;
mod timepix;
mod utility;

/// Timepix3
/// ========
///
/// Provides functions for working with Timepix3 data files
/// The function docstrings assume that `timepix` has been imported as `tpx`::
///
///   >>> import timepix3 as tpx3
///
/// Code snippets are indicated by three greater-than signs::
///
///   >>>
///
/// Works with Python 3.7 and 3.8 on Windows. Untested with other versions.
///
/// Available functions
/// -------------------
/// load
///     Loads a Timepix3 file that ends with ".tpx3" into a NumPy array
/// cluster
///     Assign cluster identifiers to each pixel activation event
#[pymodule]
fn timepix3(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Load a Timepix3, ".tpx" file into a NumPy array
    ///
    /// Parameters
    /// ----------
    /// filepath : str
    ///     Path to the ".tpx3" file that should be parsed. This parameter must
    ///     end with ".tpx3".
    ///
    /// Returns
    /// -------
    /// out : 2d ndarray of float64
    ///     Data read from the tpx3 file with each row a separate pixel and
    ///     columns 0-6:
    ///     event number | tdc | global time | tof | x | y | time over threshold
    ///
    /// Notes
    /// -----
    /// This function assumes a time-of-flight mass spectrometer was used for
    /// the generation of the Timepix3 data. It may be useful for other
    /// instruments but the return columns are most useful for TOF-MS data.
    ///
    /// Examples
    /// --------
    /// >>> import timepix3 as tpx3
    /// >>> f = "data.tpx3"
    /// >>> tpx3.load(f)
    /// >>>
    /// ...
    #[pyfn(m, "load")]
    #[text_signature = "(filepath)"]
    fn py_load(py: Python, filepath: String) -> PyResult<Py<PyArray2<f64>>> {
        let tpx_file_path = std::path::Path::new(&filepath);
        let parsed_tpx_data = timepix::parse_tpx_file(&tpx_file_path).unwrap();
        let ndarray = timepix::make_ndarray(parsed_tpx_data).unwrap();
        let pyarray = ndarray.to_pyarray(py).to_owned();
        Ok(pyarray)
    }

    /// Assign cluster identities to each event in a Timepix3 data set
    ///
    /// Parameters
    /// ----------
    /// timepix_data : 2d NumPy ndarray of float64 of size `7 x m`
    ///     Data in the same format as that produced by the `load` function with
    ///     each row a separate pixe and columns 0-6:
    ///     event number | tdc | global time | tof | x | y | time over threshold
    ///
    /// Returns
    /// -------
    /// out : 1d ndarray of int32 of length `m`
    ///     Each entry is keyed with a group number that starts counting at 0.
    ///     Entries with group numbers < 0 are noise (no cluster assignment).
    ///     Cluster numbering resets to 0 at each new event number.
    ///
    /// Notes
    /// -----
    /// Internally uses a DBSCAN function to perform density-based clustering
    /// As DBSCAN is n-squared in terms of scalable complexity, this function
    /// may take a large amount of time. Consider caching the results of this
    /// function as a saved Numpy ".npy" or ".npz" file. The example includes
    /// caching the results to a file called "cluster_labels.npy".
    #[pyfn(m, "cluster")]
    #[text_signature = "(timepix_data)"]
    fn py_cluster(
        py: Python, tpx_np_array: &PyArray2<f64>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let tpx_array = tpx_np_array.as_array();
        let tpx_vector = timepix::ndarray_to_vec(tpx_array.view());
        let cluster_labels: ndarray::Array1<i32> = Array::from(
            tpx_vector
                .par_iter()
                .map(|shot| timepix::cluster(&shot))
                .flatten()
                .collect::<Vec<i32>>(),
        );
        Ok(cluster_labels.to_pyarray(py).to_owned())
    }

    Ok(())
}
