use crate::filters::{KalmanModel, LKF};
use matrix::matrix::{funcs::inverse::identity_matrix, Matrix};
use matrix::vector::Vector;
use num_traits::Float;
use std::ops::{AddAssign, SubAssign};

impl<K: Float + SubAssign + AddAssign> LKF<K> {
    fn update_kalman_gain(&self, P_prior: &Matrix<K>, R: &Matrix<K>) -> Matrix<K> {
        let mut gain: Matrix<K> = self.H.mul_mat_ref(P_prior).mul_mat_ref(&self.H_transpose);
        gain.add_mat_ref(R);
        P_prior
            .mul_mat_ref(&self.H_transpose)
            .mul_mat_ref(&gain.inverse().unwrap())
    }

    fn update_estimation_covariance(
        &self,
        K_n: &Matrix<K>,
        R: &Matrix<K>,
        P_prior: &Matrix<K>,
    ) -> Matrix<K> {
        let P_joseph = K_n.mul_mat_ref(R).mul_mat_ref(&K_n.transpose());
        let main_part = self.I.sub_mat_ref(&K_n.mul_mat_ref(&self.H));
        main_part
            .mul_mat_ref(P_prior)
            .mul_mat_ref(&main_part.transpose())
            .add_mat_ref(&P_joseph)
    }

    fn new(
        n_x: usize,
        n_z: usize,
        n_u: usize,
        F: Matrix<K>,
        H: Matrix<K>,
        G: Option<Matrix<K>>,
    ) -> Self {
        assert_eq!(
            F.size(),
            (n_x, n_x),
            "State Transition Matrix size must be {:?}!",
            (n_x, n_x)
        );
        assert_eq!(
            H.size(),
            (n_z, n_x),
            "Observation Matrix size must be {:?}!",
            (n_z, n_x)
        );
        if let Some(g) = &G {
            assert_eq!(
                g.size(),
                (n_x, n_u),
                "Control Matrix size must be {:?}!",
                (n_x, n_u)
            );
        };
        LKF {
            n_x,
            n_z,
            n_u,
            F_transpose: F.transpose(),
            F,
            H_transpose: H.transpose(),
            H,
            G,
            I: identity_matrix(n_x),
        }
    }
}

impl<K: Float + AddAssign + SubAssign> KalmanModel<K> for LKF<K> {
    fn state_dim(&self) -> usize {
        self.n_x
    }

    fn meas_dim(&self) -> usize {
        self.n_z
    }

    fn update(
        &self,
        x_prior: &Vector<K>,
        z: &Vector<K>,
        u: &Option<Vector<K>>,
        P_prior: &Matrix<K>,
        R: &Matrix<K>,
    ) -> (Vector<K>, Matrix<K>, Matrix<K>) {
        assert_eq!(
            x_prior.size(),
            self.n_x,
            "Prior State Vector size must be {}!",
            self.n_x
        );
        assert_eq!(
            P_prior.size(),
            (self.n_x, self.n_x),
            "Prior Covariance Matrix size must be ({}, {})!",
            self.n_x,
            self.n_x
        );
        assert_eq!(
            z.size(),
            self.n_z,
            "Measurement Vector size must be {}!",
            self.n_z
        );
        assert_eq!(
            R.size(),
            (self.n_z, self.n_z),
            "Measurement Covariance Matrix size must be ({}, {})!",
            self.n_z,
            self.n_z
        );
        let K_n = self.update_kalman_gain(P_prior, R);
        let x = K_n
            .mul_vec_ref(&(z.sub_vec_ref(&self.H.mul_vec_ref(x_prior))))
            .add_vec_ref(x_prior);
        let P = self.update_estimation_covariance(&K_n, R, P_prior);
        (x, P, K_n)
    }

    fn predict(
        &mut self,
        x: &Vector<K>,
        u: &Option<Vector<K>>,
        P: &Matrix<K>,
        Q: &Matrix<K>,
    ) -> (Vector<K>, Matrix<K>) {
        assert_eq!(
            x.size(),
            self.n_x,
            "State Vector size must be {}!",
            self.n_x
        );
        assert_eq!(
            P.size(),
            (self.n_x, self.n_x),
            "Covariance Matrix size must be ({}, {})!",
            self.n_x,
            self.n_x
        );
        assert_eq!(
            Q.size(),
            (self.n_x, self.n_x),
            "Process Noise Matrix size must be ({}, {})!",
            self.n_x,
            self.n_x
        );
        let P_prior = self
            .F
            .mul_mat_ref(P)
            .mul_mat_ref(&self.F_transpose)
            .add_mat_ref(Q);
        let x_prior = self.F.mul_vec_ref(x);
        match (&self.G, u) {
            (Some(g), Some(v)) => {
                assert_eq!(
                    v.size(),
                    self.n_u,
                    "Input Vector size must be equal to {}",
                    self.n_u
                );
                (x_prior.add_vec_ref(&g.mul_vec_ref(v)), P_prior)
            }
            (Some(_), None) => {
                panic!("Control Matrix is set but Input vector is empty!")
            }
            (None, Some(_)) => {
                panic!("Control Vector provided but G is not set!");
            }
            (None, None) => (x_prior, P_prior),
        }
    }
}
