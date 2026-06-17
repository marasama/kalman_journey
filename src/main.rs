#![allow(non_snake_case)]

use kalman_2d_multivariate_xy_car::filters::{
    KalmanCore, UKF_ObservationModel, UKF_TransitionModel, UKF,
};
use matrix::{matrix::Matrix, vector::Vector};
use num_traits::Float;

fn f<K: Float>(mat: &Matrix<K>, _u: &Option<Vector<K>>) -> Matrix<K> {
    let mut new_data: Vec<K> = Vec::with_capacity(mat.rows * mat.cols);
    let delta_t = K::from(0.05).unwrap();
    // g * delta / L
    let g_L_delta_t: K = K::from(9.81 / 0.5).unwrap() * delta_t;
    let pos_row = mat.get_rows(&[0]);
    let vel_row = mat.get_rows(&[1]);
    let position: Vec<K> = pos_row
        .data
        .iter()
        .zip(vel_row.data.iter())
        .map(|(&pos, &vel)| pos + vel * delta_t)
        .collect();
    let velocity: Vec<K> = vel_row
        .data
        .iter()
        .zip(pos_row.data.iter())
        .map(|(&vel, &pos)| vel - g_L_delta_t * pos.sin())
        .collect();
    new_data.extend_from_slice(&position);
    new_data.extend_from_slice(&velocity);
    Matrix {
        data: new_data,
        rows: 2,
        cols: mat.cols,
    }
}

fn h<K: Float>(mat: &Matrix<K>) -> Matrix<K> {
    let L: K = K::from(0.5).unwrap();
    let angle_row = mat.get_rows(&[0]);
    let angles = angle_row.data.iter().map(|&x| L * x.sin()).collect();
    Matrix {
        data: angles,
        rows: 1,
        cols: mat.cols,
    }
}

use plotters::prelude::*;

fn plot_results(xs: &[f64], ys: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("state_plots.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(400);

    let time: Vec<f64> = (0..xs.len()).map(|i| i as f64).collect();

    // --- X plot ---
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut chart_x = ChartBuilder::on(&upper)
        .caption("Estimated X Position", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..time.len() as f64, x_min..x_max)?;
    chart_x
        .configure_mesh()
        .x_desc("Step")
        .y_desc("X (m)")
        .draw()?;
    chart_x.draw_series(LineSeries::new(
        time.iter().zip(xs.iter()).map(|(&t, &x)| (t, x)),
        &RED,
    ))?;
    chart_x.draw_series(
        time.iter()
            .zip(xs.iter())
            .map(|(&t, &x)| Circle::new((t, x), 2, RED.filled())),
    )?;

    // --- Y plot ---
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut chart_y = ChartBuilder::on(&lower)
        .caption("Estimated Y Position", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..time.len() as f64, y_min..y_max)?;
    chart_y
        .configure_mesh()
        .x_desc("Step")
        .y_desc("Y (m)")
        .draw()?;
    chart_y.draw_series(LineSeries::new(
        time.iter().zip(ys.iter()).map(|(&t, &y)| (t, y)),
        &BLUE,
    ))?;
    chart_y.draw_series(
        time.iter()
            .zip(ys.iter())
            .map(|(&t, &y)| Circle::new((t, y), 2, BLUE.filled())),
    )?;

    root.present()?;
    Ok(())
}

fn main() {
    // Array of radar measurement tuples: (range in meters, bearing in radians)
    let meas = [
        0.119, 0.113, 0.12, 0.101, 0.099, 0.063, 0.008, -0.017, -0.037, -0.05,
    ];
    const DELTA_T: f64 = 0.05;
    let a = UKF::new(
        2,
        1,
        0,
        UKF_TransitionModel::NonLinear { f },
        UKF_ObservationModel::NonLinear { h },
        Option::None,
        0.1,
        2.,
        0.,
    );
    let Q = Matrix::from([[1.5625e-6, 6.25e-5], [6.25e-5, 0.0025]]);

    let R: Matrix<f64> = Matrix::from([[0.01.powf(2.)]]);

    let P: Matrix<f64> = Matrix::from([[5., 0.], [0., 5.]]);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut b = KalmanCore::new(a, Vector::from([0.0873, 0.]), P, Q, R);

    for (i, &val) in meas.iter().enumerate() {
        b.step(&Vector::from([val]), &Option::None);
        let state = b.state();
        xs.push(state.data[0]); // x position
        ys.push(state.data[1]); // y position
        println!("------------{}-------------", i + 1);
        println!("State: {:.6}", b.state());
        println!("Covariance: {:.6}", b.covariance());
    }
    plot_results(&xs, &ys);
}
