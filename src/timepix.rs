use crate::{dbscan, utility::euclidean_distance};

use itertools::Itertools;
use ndarray as nd;
use ndarray_stats::QuantileExt;
use rayon::prelude::*;
use std::convert::TryInto;

#[derive(Debug, Copy, Clone)]
pub struct TPXPoint {
    pub shot_count: f64,
    pub tdc: f64,
    pub global_time: f64,
    pub tof: f64,
    pub x: f64,
    pub y: f64,
    pub tot_ns: f64,
}

impl TPXPoint {
    pub fn from_row(row: nd::ArrayView1<f64>) -> TPXPoint {
        TPXPoint {
            shot_count: row[0],
            tdc: row[1],
            global_time: row[2],
            tof: row[3],
            x: row[4],
            y: row[5],
            tot_ns: row[6],
        }
    }
}


/// Parses a TPX3 data file by packets of 8 bytes and calculate a mass
/// spectrometry time-of-flight parameter.
pub fn parse_tpx_file(
    tpx_file_path: &std::path::Path,
) -> std::io::Result<Vec<Vec<TPXPoint>>> {
    let raw_data = std::fs::read(tpx_file_path)?;
    assert!(raw_data.len() % 8 == 0);
    let (mut tdc, mut new_tdc, mut shot_count, mut row) = (0.0, true, 0, 0);
    let mut tpx_data = vec![];
    for packet in raw_data.chunks(8) {
        let int_packet = u64::from_le_bytes(
            packet.try_into().expect("slice with incorrect length"),
        );

        // Header - don't do anything
        if &packet[..4] == b"TPX3" {
            {};

        // TDC entry
        } else if int_packet >> 60 == 0x6 {
            let coarsetime = ((int_packet >> 12) & 0xFFFF_FFFF) as f64 * 25E-9;
            let finetime =
                (((int_packet >> 5) & 0xF).wrapping_sub(1) << 9) as u64 / 12;
            let fttime = (int_packet & 0x0000_0000_0000_0E00)
                | (finetime & 0x0000_0000_0000_01FF);
            tdc = coarsetime + fttime as f64 * (25.0 / 4096.0) * 1E-9;
            new_tdc = true; // Reset packet data flag

        // Data point entry
        } else if int_packet >> 60 == 0xb {
            if new_tdc {
                tpx_data.push(vec![]);
                shot_count += 1;
                new_tdc = false;
            }
            row += 1;
            let coarse_toa = ((((int_packet >> 30) & 0x3FFF) << 4)
                | (!((int_packet >> 16) & 0xF) & 0xF))
                as f64;
            let global_time = ((int_packet & 0xFFFF) * 25 * 16384) as f64
                + coarse_toa * (25.0 / 16.0);
            let mut tof = global_time / 1E9 - tdc;
            let clock_align = 107.374_182_4 / 4.0; // for clock reset cycles
            if tof < 0.0 { // time-of-flight parameter for mass spectrometry
                tof += (tof.abs() / clock_align).floor() * (clock_align);
            }
            let pix = (int_packet & 0x0000_7000_0000_0000) >> 44;
            let x = ((int_packet & 0x0FE0_0000_0000_0000) >> 52) + pix / 4;
            let y = ((int_packet & 0x001F_8000_0000_0000) >> 45) + (pix & 0x3);
            let tot_ns = ((int_packet >> 20) & 0x3FF) * 25;
            tpx_data[shot_count - 1].push(TPXPoint {
                shot_count: (shot_count as f64) - 1.0,
                tdc,
                global_time,
                tof,
                x: x as f64, // setting everything as f64 makes it easier
                y: y as f64, // to export, although it expands the file size
                tot_ns: tot_ns as f64,
            });
        }
    }
    Ok(tpx_data)
}


/// Makes an ndarray from tpx_data
pub fn make_ndarray(
    tpx_data: Vec<Vec<TPXPoint>>,
) -> Result<ndarray::Array2<f64>, Box<dyn std::error::Error>> {
    let mut new_vec = vec![];
    for shot in tpx_data.iter() {
        for tpxpoint in shot.iter() {
            new_vec.push(tpxpoint.shot_count);
            new_vec.push(tpxpoint.tdc);
            new_vec.push(tpxpoint.global_time);
            new_vec.push(tpxpoint.tof);
            new_vec.push(tpxpoint.x);
            new_vec.push(tpxpoint.y);
            new_vec.push(tpxpoint.tot_ns);
        }
    }
    let rows = new_vec.len() / 7;
    let array: ndarray::Array2<f64> =
        ndarray::Array::from_shape_vec((rows, 7), new_vec)?;
    Ok(array)
}


/// convert an ndarray to a vec of vecs -> helper function
pub fn ndarray_to_vec(array: nd::ArrayView2<f64>) -> Vec<Vec<TPXPoint>> {
    let number_of_rows = *array.column(0).max().unwrap() as usize + 1;
    let mut tpx_data = vec![vec![]; number_of_rows];
    for row in array.genrows() {
        tpx_data[row[0] as usize].push(TPXPoint::from_row(row));
    }
    tpx_data
}


/// Performs a splitting of timepix points to run DBSCAN on (see dbscan.rs) file
/// This splitting is done by the time-of-flight (tof) parameter and is intended
/// for tof-mass spectrometry data
/// Takes a slice of TPXPoints and returns a vec of labels:
/// a label of -1 is a noise point
/// Edge (if any remain) and core points are simply labeled by an increasing
/// integer
pub fn cluster(shot_data: &[TPXPoint]) -> Vec<i32> {
    let min_pts = 2;
    let eps = 2.0; // epsilon for dbscan and distance for preprocessing
    // scalar factor for adjusting the tof to be the same "distance" as x and y 
    let time_factor = eps / (81_920.0 * (25.0 / 4096.0) * 1.0E-9);

    // In order to speed up dbscan, it is helpful to preprocess the data into
    // smaller chunks by their tof. To chunk by tof > eps, the data should be
    // sorted. In order to regain the data's original order, the original
    // indices need to be calculated and used.
    let seq_points = shot_data // sequential, clusterable points
        .iter()
        .enumerate()
        .map(|(i, row)| [i as f64, row.tof * time_factor, row.x, row.y])
        .sorted_by(|a, b| a[1].partial_cmp(&b[1]).expect("NaN encountered"))
        .collect::<Vec<_>>();
    // split the sequential points into new vectors when tof > eps
    let initial_point = seq_points[0][1..=3].to_vec();
    let mut split_points = vec![vec![initial_point]]; // ugly 3-dimensional vec
    for i in 0..(seq_points.len() - 1) {
        let point = seq_points[i + 1][1..=3].to_vec();
        if (point[0] - seq_points[i][1]) > eps {
            split_points.push(vec![point]);
        } else {
            split_points.last_mut().unwrap().push(point)
        }
    }
    // dbscan the split data and combine labels by counting to avoid duplicates
    let (mut group_labels, mut label_counter) = (vec![], 0);
    for group in split_points.iter() {
        let labels = dbscan::dbscan(group, eps, min_pts, euclidean_distance)
            .into_iter()
            .map(|label| match label {
                dbscan::Label::Core(grp_num) => grp_num as i32 + label_counter,
                dbscan::Label::Edge(grp_num) => grp_num as i32 + label_counter,
                _ => -1, // is a noise point and so is just labeled -1
            })
            .collect::<Vec<i32>>();
        label_counter = match labels.iter().max() {
            Some(-1) => label_counter, // only noise, ignore value
            Some(value) => value + 1, // keep counting, store for next iteration
            None => panic!["No labels found in data."], // TODO: deal with this
            // case instead of panicking
        };
        group_labels.push(labels);
    }
    // flatten and sort by the (zipped) indices from seq_points
    group_labels
        .iter()
        .flatten()
        .zip(seq_points.iter().map(|x| x[0] as i32)) // (quick calc of indices)
        .sorted_by(|(_, i), (_, i_next)| i.cmp(&i_next)) // sort by indices
        .map(|(&group_label, _)| group_label) // throw away indices
        .collect()
}