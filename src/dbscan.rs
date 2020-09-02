use rayon::prelude::*;

/// Classification or Label according to the DBSCAN algorithm
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Label {
    Core(usize),
    Edge(usize),
    Noise,
    Unchecked,
}

/// The DBSCAN algorithm
///
/// # Arguments
/// * `database` - a Vec<Vec<f64>> "DB" or database, organized by row
/// * `eps` - maximum distance between datapoints within a cluster
/// * `min_points` - minimum number of datapoints to make a cluster
/// * `dist_func` - a distance function
#[inline]
pub fn dbscan<T>(
    dataset: &[Vec<T>], eps: f64, min_points: usize,
    dist_func: fn(&[T], &[T]) -> f64,
) -> Vec<Label>
where
    T: Copy + Sync,
    f64: From<T>,
{
    let mut current_cluster = 0;
    let mut labels = vec![Label::Unchecked; dataset.len()];
    for (i, point) in dataset.iter().enumerate() {
        if labels[i] != Label::Unchecked {
            continue;
        }
        let mut neighbors = region_query(dataset, point, eps, dist_func);
        if neighbors.len() < min_points {
            labels[i] = Label::Noise;
        } else {
            labels[i] = Label::Core(current_cluster);
            let mut j = 0;
            while j < neighbors.len() {
                let neighbor_point = neighbors[j];
                labels[neighbor_point] = match labels[neighbor_point] {
                    Label::Noise => Label::Edge(current_cluster),
                    Label::Unchecked => {
                        let mut new_neighbors = region_query(
                            dataset,
                            &dataset[neighbor_point],
                            eps,
                            dist_func,
                        );
                        if new_neighbors.len() >= min_points {
                            neighbors.append(&mut new_neighbors);
                        };
                        Label::Core(current_cluster)
                    }
                    _ => labels[neighbor_point],
                };
                j += 1;
            }
            current_cluster += 1;
        }
    }
    labels
}

/// A sub-function of DBSCAN that is called in two different places to filter
/// points that are outside of the eps range of the distance function
#[inline]
fn region_query<T>(
    dataset: &[Vec<T>], point: &[T], eps: f64, distfn: fn(&[T], &[T]) -> f64,
) -> Vec<usize>
where
    T: Copy + Sync,
    f64: From<T>,
{
    dataset
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if distfn(point, p) < eps {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}
