use crate::filters::{EKF_ObservationModel, EKF_TransitionModel, KalmanModel, EKF};
use matrix::matrix::{funcs::inverse::identity_matrix, Matrix};
use matrix::vector::Vector;
use num_traits::Float;
use std::ops::{AddAssign, SubAssign};

impl<K: Float + AddAssign + SubAssign> EKF<K> {
    fn update_kalman_gain(
        &self,
        x_prior: &Vector<K>,
        u: &Option<Vector<K>>,
        P_prior: &Matrix<K>,
        R: &Matrix<K>,
    ) -> Matrix<K> {
        match &self.H {
            EKF_ObservationModel::Linear { H, H_transpose } => {
                let mut tmp = H
                    .mul_mat_ref(P_prior)
                    .mul_mat_ref(H_transpose)
                    .add_mat_ref(R);
                P_prior
                    .mul_mat_ref(H_transpose)
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
            EKF_ObservationModel::NonLinear { jac, .. } => {
                assert_eq!(
                    x_prior.size(),
                    self.n_x,
                    "Prior State Vector size must be {}!",
                    self.n_x
                );
                let jacobian = jac(x_prior, u);
                let mut tmp = jacobian
                    .mul_mat_ref(P_prior)
                    .mul_mat_ref(&jacobian.transpose())
                    .add_mat_ref(R);
                P_prior
                    .mul_mat_ref(&jacobian.transpose())
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
        }
    }

    fn update_estimation_covariance(
        &self,
        x_prior: &Vector<K>,
        u: &Option<Vector<K>>,
        K_n: &Matrix<K>,
        R: &Matrix<K>,
        P_prior: &Matrix<K>,
    ) -> Matrix<K> {
        let joseph = K_n.mul_mat_ref(R).mul_mat_ref(&K_n.transpose());
        let main_part: Matrix<K> = match &self.H {
            EKF_ObservationModel::Linear { H, .. } => self.I.sub_mat_ref(&K_n.mul_mat_ref(H)),
            EKF_ObservationModel::NonLinear { jac, .. } => {
                self.I.sub_mat_ref(&K_n.mul_mat_ref(&jac(x_prior, u)))
            }
        };
        main_part
            .mul_mat_ref(P_prior)
            .mul_mat_ref(&main_part.transpose())
            .add_mat_ref(&joseph)
    }

    pub fn new(
        n_x: usize,
        n_z: usize,
        n_u: usize,
        F: EKF_TransitionModel<K>,
        H: EKF_ObservationModel<K>,
        G: Option<Matrix<K>>,
    ) -> Self {
        match &F {
            EKF_TransitionModel::NonLinear { .. } => println!(
                "Be aware: Transition f(x) and jac(x) must return {} size vector and {:?} size matrix!",
                n_x, (n_x, n_x)
            ),
            EKF_TransitionModel::Linear { F, .. } => assert_eq!(
                F.size(),
                (n_x, n_x),
                "State Transition Matrix size must be {:?}!",
                (n_x, n_x)
            ),
        }

        match &H {
            EKF_ObservationModel::NonLinear { .. } => println!(
                "Be aware: Observation f(x) and jac(x) must return {} size vector and {:?} size matrix!",
                n_x, (n_z, n_x)
            ),
            EKF_ObservationModel::Linear { H, .. } => assert_eq!(
                H.size(),
                (n_z, n_x),
                "Observation Matrix size must be {:?}!",
                (n_z, n_x)
            ),
        }

        println!("Observation Matrix size must be (n_z, n_x)!");
        if matches!(F, EKF_TransitionModel::NonLinear { .. }) && G.is_some() {
            panic!("G is set but Nonlinear Transition Model does not use it!");
        }
        EKF {
            n_x,
            n_z,
            n_u,
            F,
            H,
            G,
            I: identity_matrix(n_x),
        }
    }
}

impl<K: Float + SubAssign + AddAssign> KalmanModel<K> for EKF<K> {
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
        let K_n = self.update_kalman_gain(x_prior, u, P_prior, R);
        let x = match &self.H {
            EKF_ObservationModel::Linear { H, .. } => K_n
                .mul_vec_ref(&(z.sub_vec_ref(&H.mul_vec_ref(x_prior))))
                .add_vec_ref(x_prior),
            EKF_ObservationModel::NonLinear { f, .. } => K_n
                .mul_vec_ref(&(z.sub_vec_ref(&f(x_prior, u))))
                .add_vec_ref(x_prior),
        };
        let P = self.update_estimation_covariance(x_prior, u, &K_n, R, P_prior);

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

        match &self.F {
            EKF_TransitionModel::Linear { F, F_transpose } => {
                let x_prior = F.mul_vec_ref(x);
                let P_prior = F.mul_mat_ref(P).mul_mat_ref(F_transpose).add_mat_ref(Q);
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
            EKF_TransitionModel::NonLinear { f, jac } => {
                let jacobian = jac(x, u);
                (
                    f(x, u),
                    jacobian
                        .mul_mat_ref(P)
                        .mul_mat_ref(&jacobian.transpose())
                        .add_mat_ref(Q),
                )
            }
        }
    }
}
