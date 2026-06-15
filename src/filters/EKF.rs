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
        f: fn(&Vector<K>, Option<&Vector<K>>) -> Vector<K>,
        jac: fn(&Vector<K>, Option<&Vector<K>>) -> Matrix<K>,
    },
}

enum ObservationModel<K: Float> {
    Linear {
        H: Matrix<K>,
        H_transpose: Matrix<K>,
    },
    Nonlinear {
        f: fn(&Vector<K>, Option<&Vector<K>>) -> Vector<K>,
        jac: fn(&Vector<K>, Option<&Vector<K>>) -> Matrix<K>,
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

impl<K: Float + SubAssign + AddAssign> EKF<K> {
    fn update_kalman_gain(
        &self,
        x_prior: &Vector<K>,
        u: &Vector<K>,
        P_prior: &Matrix<K>,
        R: &Matrix<K>,
    ) -> Matrix<K> {
        match &self.H {
            ObservationModel::Linear { H, H_transpose } => {
                let mut tmp = H.mul_mat_ref(P_prior).mul_mat_ref(H_transpose).add_ref(R);
                P_prior
                    .mul_mat_ref(H_transpose)
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
            ObservationModel::Nonlinear { f, jac } => {
                assert_eq!(
                    x_prior.size(),
                    self.n_x,
                    "Prior State Vector size must be {}!",
                    self.n_x
                );
                let jacobian = jac(x_prior, Some(u));
                let mut tmp = jacobian
                    .mul_mat_ref(P_prior)
                    .mul_mat_ref(&jacobian.transpose())
                    .add_ref(R);
                P_prior
                    .mul_mat_ref(&jacobian.transpose())
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
        }
    }

    fn update_estimation_covariance(
        &self,
        x_prior: &Vector<K>,
        u: &Vector<K>,
        K_n: &Matrix<K>,
        R: &Matrix<K>,
        P_prior: &Matrix<K>,
    ) -> Matrix<K> {
        let joseph = K_n.mul_mat_ref(R).mul_mat_ref(&K_n.transpose());
        let main_part: Matrix<K> = match &self.H {
            ObservationModel::Linear { H, H_transpose } => self.I.sub_ref(&K_n.mul_mat_ref(H)),
            ObservationModel::Nonlinear { f, jac } => {
                self.I.sub_ref(&K_n.mul_mat_ref(&jac(x_prior, Some(u))))
            }
        };
        main_part
            .mul_mat_ref(P_prior)
            .mul_mat_ref(&main_part.transpose())
            .add_ref(&joseph)
    }
}
