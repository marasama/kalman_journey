#![allow(non_snake_case)]
#![allow(unused)]
pub mod EKF;
pub mod LKF;

use matrix::matrix::Matrix;
use matrix::vector::Vector;
use num_traits::Float;

trait KalmanModel<K: Float> {
    fn state_dim(&self) -> usize;
    fn meas_dim(&self) -> usize;
    fn predict(
        &self,
        x: &Vector<K>,
        u: &Vector<K>,
        P: &Matrix<K>,
        Q: &Matrix<K>,
    ) -> (Vector<K>, Matrix<K>); // x_prior, P_prior
    fn update(
        &self,
        x_prior: &Vector<K>,
        P_prior: &Matrix<K>,
        z: &Vector<K>,
        R: &Matrix<K>,
    ) -> (Vector<K>, Matrix<K>, Matrix<K>); // x, P, K
}

struct KalmanCore<K: Float, M: KalmanModel<K>> {
    model: M,
    x: Vector<K>,
    P: Matrix<K>,
    K: Matrix<K>,
    Q: Matrix<K>,
    R: Matrix<K>,
}

impl<K: Float, M: KalmanModel<K>> KalmanCore<K, M> {
    fn state(&self) -> &Vector<K> {
        &self.x
    }

    fn covariance(&self) -> &Matrix<K> {
        &self.P
    }

    fn kalman_gain(&self) -> &Matrix<K> {
        &self.K
    }

    fn step(&mut self, z: &Vector<K>, u: &Vector<K>) {
        let (x_prior, P_prior) = self.model.predict(&self.x, u, &self.P, &self.Q);
        (self.x, self.P, self.K) = self.model.update(&x_prior, &P_prior, z, &self.R);
    }
}
