#![allow(non_snake_case)]
#![allow(unused)]
pub mod EKF_fn;
pub mod LKF_fn;
pub mod UKF_fn;

use matrix::matrix::Matrix;
use matrix::vector::Vector;
use num_traits::Float;

pub trait KalmanModel<K: Float> {
    fn state_dim(&self) -> usize;
    fn meas_dim(&self) -> usize;
    fn predict(
        &mut self,
        x: &Vector<K>,
        u: &Option<Vector<K>>,
        P: &Matrix<K>,
        Q: &Matrix<K>,
    ) -> (Vector<K>, Matrix<K>); // x_prior, P_prior
    fn update(
        &self,
        x_prior: &Vector<K>,
        z: &Vector<K>,
        u: &Option<Vector<K>>,
        P_prior: &Matrix<K>,
        R: &Matrix<K>,
    ) -> (Vector<K>, Matrix<K>, Matrix<K>); // x, P, K
}

pub struct KalmanCore<K: Float, M: KalmanModel<K>> {
    pub model: M,
    pub x: Vector<K>,
    pub P: Matrix<K>,
    K: Matrix<K>,
    pub Q: Matrix<K>,
    pub R: Matrix<K>,
}

impl<K: Float, M: KalmanModel<K>> KalmanCore<K, M> {
    pub fn state(&self) -> &Vector<K> {
        &self.x
    }

    pub fn covariance(&self) -> &Matrix<K> {
        &self.P
    }

    pub fn kalman_gain(&self) -> &Matrix<K> {
        &self.K
    }

    pub fn step(&mut self, z: &Vector<K>, u: &Option<Vector<K>>) {
        let (x_prior, P_prior) = self.model.predict(&self.x, u, &self.P, &self.Q);
        (self.x, self.P, self.K) = self.model.update(&x_prior, z, u, &P_prior, &self.R);
    }

    pub fn new(model: M, x: Vector<K>, P: Matrix<K>, Q: Matrix<K>, R: Matrix<K>) -> Self {
        let n_x = model.state_dim();
        let n_z = model.meas_dim();
        assert_eq!(x.size(), n_x, "State Vector size must be equal to {}!", n_x,);
        assert_eq!(
            P.size(),
            (n_x, n_x),
            "Covariance Matrix size must be equal to {:?}!",
            (n_x, n_x)
        );
        assert_eq!(
            Q.size(),
            (n_x, n_x),
            "Process Noise Matrix size must be equal to {:?}!",
            (n_x, n_x)
        );
        assert_eq!(
            R.size(),
            (n_z, n_z),
            "Measurement Noise Matrix size must be equal to {:?}!",
            (n_z, n_z)
        );

        Self {
            model,
            x,
            P,
            K: Matrix::empty(),
            Q,
            R,
        }
    }
}

// Unscented Kalman Filter -----------------------------------------------
pub enum UKF_TransitionModel<K: Float> {
    Linear {
        F: Matrix<K>,
    },
    NonLinear {
        f: fn(&Matrix<K>, &Option<Vector<K>>) -> Matrix<K>,
    },
}

pub enum UKF_ObservationModel<K: Float> {
    Linear { H: Matrix<K> },
    NonLinear { h: fn(&Matrix<K>) -> Matrix<K> },
}

/// Unscented Kalman Filter
pub struct UKF<K: Float> {
    /// State Vector
    pub n_x: usize,
    /// Measurement Vector
    pub n_z: usize,
    /// Input Vector
    pub n_u: usize,
    /// Transition Model
    F: UKF_TransitionModel<K>,
    /// Observation Model
    H: UKF_ObservationModel<K>,
    /// Control Matrix
    G: Option<Matrix<K>>,
    /// Sigma Points of the next step
    sigma_points_prior: Matrix<K>,
    /// Lambda calculated with tune parameters for calculating the sigma points
    lambda: K,
    /// Weights for calculating the sigma points
    weights: Vector<K>,
    /// Weight but in diagonal for computational advantage
    weights_diag: Matrix<K>,
}

// Extended Kalman Filter --------------------------------------------------
pub enum EKF_TransitionModel<K: Float> {
    Linear {
        F: Matrix<K>,
        F_transpose: Matrix<K>,
    },
    NonLinear {
        f: fn(&Vector<K>, &Option<Vector<K>>) -> Vector<K>,
        jac: fn(&Vector<K>, &Option<Vector<K>>) -> Matrix<K>,
    },
}

pub enum EKF_ObservationModel<K: Float> {
    Linear {
        H: Matrix<K>,
        H_transpose: Matrix<K>,
    },
    NonLinear {
        f: fn(&Vector<K>, &Option<Vector<K>>) -> Vector<K>,
        jac: fn(&Vector<K>, &Option<Vector<K>>) -> Matrix<K>,
    },
}

/// Extended Kalman Filter
pub struct EKF<K: Float> {
    n_x: usize,
    n_z: usize,
    n_u: usize,
    F: EKF_TransitionModel<K>,
    H: EKF_ObservationModel<K>,
    G: Option<Matrix<K>>,
    I: Matrix<K>,
}

// Linear Kalman Filter --------------------------------------------------
/// Linear Kalman Filter
pub struct LKF<K: Float> {
    n_x: usize,
    n_z: usize,
    n_u: usize,
    F: Matrix<K>,
    F_transpose: Matrix<K>,
    H: Matrix<K>,
    H_transpose: Matrix<K>,
    G: Option<Matrix<K>>,
    I: Matrix<K>,
}
