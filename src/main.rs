#![allow(non_snake_case)]

use kalman_2d_multivariate_xy_car::filters::{
    KalmanCore, KalmanModel, UKF_ObservationModel, UKF_TransitionModel, UKF,
};
use matrix::{matrix::Matrix, vector::Vector};
use num_traits::Float;

fn h<K: Float>(mat: &Matrix<K>) -> Matrix<K> {
    let x_row = mat.get_rows(&[0]);
    let y_row = mat.get_rows(&[3]);

    let mut new_data = Vec::with_capacity(mat.cols * 2);

    let range: Vec<K> = x_row
        .data
        .iter()
        .zip(y_row.data.iter())
        .map(|(&x, &y)| (num_traits::pow(x, 2) + num_traits::pow(y, 2)).sqrt())
        .collect();

    let bearing: Vec<K> = x_row
        .data
        .iter()
        .zip(y_row.data.iter())
        .map(|(&x, &y)| y.atan2(x))
        .collect();

    new_data.extend_from_slice(&range);
    new_data.extend_from_slice(&bearing);
    Matrix {
        data: new_data,
        rows: 2,
        cols: mat.cols,
    }
}

fn main() {
    // Array of radar measurement tuples: (range in meters, bearing in radians)
    let meas: [(f64, f64); 35] = [
        (502.55, -0.9316),
        (477.34, -0.8977),
        (457.21, -0.8512),
        (442.94, -0.8114),
        (427.27, -0.785),
        (406.05, -0.7392),
        (400.73, -0.7052),
        (377.32, -0.6478),
        (360.27, -0.59),
        (345.93, -0.5183),
        (333.34, -0.4698),
        (328.07, -0.3952),
        (315.48, -0.3026),
        (301.41, -0.2445),
        (302.87, -0.1626),
        (304.25, -0.0937),
        (294.46, 0.0085),
        (294.29, 0.0856),
        (299.38, 0.1675),
        (299.37, 0.2467),
        (300.68, 0.329),
        (304.1, 0.4149),
        (301.96, 0.504),
        (300.3, 0.5534), // Checked closely: 0.5934 in row 24
        (301.9, 0.667),
        (296.7, 0.7537),
        (297.07, 0.8354),
        (295.29, 0.9195),
        (296.31, 1.0039),
        (300.62, 1.0923),
        (292.3, 1.1546),
        (298.11, 1.2564),
        (298.07, 1.3274),
        (298.92, 1.409),
        (298.04, 1.5011),
    ];
    const DELTA_T: f64 = 1.;
    let F: Matrix<f64> = Matrix::from([
        [1., DELTA_T, 0.5 * DELTA_T.powf(2.), 0., 0., 0.],
        [0., 1., DELTA_T, 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., DELTA_T, 0.5 * DELTA_T.powf(2.)],
        [0., 0., 0., 0., 1., DELTA_T],
        [0., 0., 0., 0., 0., 1.],
    ]);
    let a = UKF::new(
        6,
        2,
        0,
        UKF_TransitionModel::Linear { F },
        UKF_ObservationModel::NonLinear { h },
        Option::None,
        -3.,
    );
    let P = Matrix::from([
        [500., 0., 0., 0., 0., 0.],
        [0., 500., 0., 0., 0., 0.],
        [0., 0., 500., 0., 0., 0.],
        [0., 0., 0., 500., 0., 0.],
        [0., 0., 0., 0., 500., 0.],
        [0., 0., 0., 0., 0., 500.],
    ]);

    let Q = Matrix::from([
        [
            DELTA_T.powf(4.) / 4.,
            DELTA_T.powf(3.) / 2.,
            DELTA_T.powf(2.) / 2.,
            0.,
            0.,
            0.,
        ],
        [DELTA_T.powf(3.) / 2., DELTA_T.powf(2.), DELTA_T, 0., 0., 0.],
        [DELTA_T.powf(2.) / 2., DELTA_T, 1., 0., 0., 0.],
        [
            0.,
            0.,
            0.,
            DELTA_T.powf(4.) / 4.,
            DELTA_T.powf(3.) / 2.,
            DELTA_T.powf(2.),
        ],
        [0., 0., 0., DELTA_T.powf(3.) / 2., DELTA_T.powf(2.), DELTA_T],
        [0., 0., 0., DELTA_T.powf(2.) / 2., DELTA_T, 1.],
    ]);

    let R = Matrix::from([[5f64.powf(2.), 0.], [0., 0.0087.powf(2.)]]);

    let mut b = KalmanCore::new(a, Vector::from([400., 0., 0., -300., 0., 0.]), P, Q, R);

    for (i, &val) in meas.iter().enumerate() {
        b.step(&Vector::from([val.0, val.1]), &Option::None);
        println!("------------{}-------------", i + 1);
        println!("State: {}", b.state());
        println!("Covariance: {}", b.covariance());
    }
}
