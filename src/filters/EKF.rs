use crate::filters::KalmanModel;
use matrix::matrix::Matrix;
use matrix::vector::Vector;
use num_traits::Float;
use std::ops::{AddAssign, SubAssign};

enum TransitionModel<K: Float> {
    Linear {
        F: Matrix<K>,
        F_transpose: Matrix<K>,
    },
    Nonlinear {
        f: fn(&Vector<K>, Option<Vector<K>>) -> Vector<K>,
        jac: fn(&Vector<K>, Option<Vector<K>>) -> Matrix<K>,
    },
}

enum ObservationModel<K: Float> {
    Linear {
        H: Matrix<K>,
        H_transpose: Matrix<K>,
    },
    Nonlinear {
        f: fn(&Vector<K>, Option<Vector<K>>) -> Vector<K>,
        jac: fn(&Vector<K>, Option<Vector<K>>) -> Matrix<K>,
    },
}
struct EKF<K: Float> {
    n_x: usize,
    n_z: usize,
    n_u: usize,
    F: TransitionModel<K>,
    H: ObservationModel<K>,
    G: Option<Matrix<K>>,
    I: Matrix<K>,
}
