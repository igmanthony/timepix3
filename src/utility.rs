/// A simple progress bar to indicate progress
pub fn progressbar(iteration: usize, total: usize, message: &str) {
    let percent = 100.0 * (iteration as f64 / total as f64);
    let filled_length = 30 * (iteration + 1) / total;
    let prog_bar = "#".repeat(filled_length) + &"=".repeat(30 - filled_length);
    print!("{} [{}] {:.1} %\r", message, prog_bar, percent);
}

/// Euclidean distance between two vectors of points
#[inline]
pub fn euclidean_distance<T>(a: &[T], b: &[T]) -> f64
where
    f64: From<T>,
    T: Copy,
{
    a.iter()
        .zip(b.iter())
        .fold(0.0, |acc, (&x, &y)| {
            acc + (f64::from(x) - f64::from(y)).powi(2)
        })
        .sqrt()
}
