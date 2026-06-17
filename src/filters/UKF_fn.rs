use crate::filters::{KalmanModel, UKF_ObservationModel, UKF_TransitionModel, UKF};
use matrix::matrix::{funcs::inverse::identity_matrix, Matrix};
use matrix::vector::Vector;
use num_traits::Float;
use std::ops::{AddAssign, Mul, SubAssign};

impl<K: Float + AddAssign + SubAssign> UKF<K> {
    fn calculate_sigma_points(&self, x: &Vector<K>, P: &Matrix<K>) -> Matrix<K> {
        let mut data: Vec<K> = vec![K::zero(); (self.n_x * 2 + 1) * self.n_x];
        let scale: K = K::from(self.n_x).unwrap() + self.lambda;
        let num_sigma_points = self.n_x * 2 + 1;
        let sqrt_mat = P.scl_mat_ref(scale).cholesky();
        for r in 0..self.n_x {
            // First columns is state vector itself
            data[r * num_sigma_points] = x.data[r];
            for i in 0..self.n_x {
                let delta = sqrt_mat.data[r * self.n_x + i];
                let col_positive = 1 + i;
                let col_negative = 1 + self.n_x + i;

                data[r * num_sigma_points + col_positive] = x.data[r] + delta;
                data[r * num_sigma_points + col_negative] = x.data[r] - delta;
            }
        }

        Matrix {
            data,
            rows: self.n_x,
            cols: num_sigma_points,
        }
    }

    fn calculate_weights(w_0: K, w_0_c: K, w_i: K, length: usize) -> (Vector<K>, Matrix<K>) {
        let mut diagonal = identity_matrix(length);
        diagonal.scl(w_i);
        diagonal.data[0] = w_0_c;
        let mut new_data = vec![w_i; length];
        new_data[0] = w_0;
        (Vector { data: new_data }, diagonal)
    }

    pub fn new(
        n_x: usize,
        n_z: usize,
        n_u: usize,
        F: UKF_TransitionModel<K>,
        H: UKF_ObservationModel<K>,
        G: Option<Matrix<K>>,
        alpha: K,
        beta: K,
        kappa: K,
    ) -> Self {
        let n = K::from(n_x).unwrap();
        match &F {
            UKF_TransitionModel::Linear { F } => assert_eq!(
                F.size(),
                (n_x, n_x),
                "State Transition Matrix size must be {:?}!",
                (n_x, n_x)
            ),
            UKF_TransitionModel::NonLinear { .. } => println!(
                "Be aware: Transition f(x) must return {:?} size matrix!",
                (n_x, n_x * 2 + 1)
            ),
        }

        match &H {
            UKF_ObservationModel::Linear { H } => assert_eq!(
                H.size(),
                (n_z, n_x),
                "Observation Matrix size must be {:?}!",
                (n_z, n_x)
            ),
            UKF_ObservationModel::NonLinear { .. } => println!(
                "Be aware: Observation f(x) must return {:?} size matrix!",
                (n_z, n_x * 2 + 1)
            ),
        }

        if matches!(F, UKF_TransitionModel::NonLinear { .. }) && G.is_some() {
            panic!("G is set but Nonlinear Transition Model does not use it!");
        }

        let lambda = num_traits::pow(alpha, 2) * (n + kappa) - n;
        let (weights, weights_diag) = UKF::calculate_weights(
            lambda / (n + lambda),
            lambda / (n + lambda) + (K::one() - num_traits::pow(alpha, 2) + beta),
            K::one() / (K::from(2).unwrap() * (n + lambda)),
            2 * n_x + 1,
        );

        UKF {
            n_x,
            n_z,
            n_u,
            F,
            H,
            G,
            sigma_points_prior: Matrix::empty(),
            lambda,
            weights,
            weights_diag,
        }
    }
}

impl<K: Float + AddAssign + SubAssign> KalmanModel<K> for UKF<K> {
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
        let meas_space = match &self.H {
            UKF_ObservationModel::Linear { H } => H.mul_mat_ref(&self.sigma_points_prior),
            UKF_ObservationModel::NonLinear { h } => h(&self.sigma_points_prior),
        };

        let pred_meas_mean = meas_space.mul_vec_ref(&self.weights);
        let tmp_subt = meas_space.sub_each_col(&pred_meas_mean);
        let meas_space_cov = tmp_subt
            .mul_mat_ref(&self.weights_diag)
            .mul_mat_ref(&tmp_subt.transpose())
            .add_mat_ref(R);
        let cross_cov_state_meas = self
            .sigma_points_prior
            .sub_each_col(x_prior)
            .mul_mat_ref(&self.weights_diag)
            .mul_mat_ref(&tmp_subt.transpose());
        let K_n = cross_cov_state_meas.mul_mat_ref(&meas_space_cov.inverse().unwrap());
        let x = x_prior.add_vec_ref(&K_n.mul_vec_ref(&z.sub_vec_ref(&pred_meas_mean)));
        let P = P_prior.sub_mat_ref(
            &K_n.mul_mat_ref(&meas_space_cov)
                .mul_mat_ref(&K_n.transpose()),
        );
        (x, P, K_n)
    }
    fn predict(
        &mut self,
        x: &Vector<K>,
        u: &Option<Vector<K>>,
        P: &Matrix<K>,
        Q: &Matrix<K>,
    ) -> (Vector<K>, Matrix<K>) {
        let sigma_points = self.calculate_sigma_points(x, P);
        self.sigma_points_prior = match &self.F {
            UKF_TransitionModel::Linear { F, .. } => {
                let mut transformed = F.mul_mat_ref(&sigma_points);
                if let (Some(g), Some(v)) = (&self.G, u) {
                    assert_eq!(
                        v.size(),
                        self.n_u,
                        "Input Vector size must be equal to {}",
                        self.n_u
                    );
                    transformed = transformed.add_each_col(&g.mul_vec_ref(v))
                } else if self.G.is_some() || u.is_some() {
                    panic!("Control Vector or Input Variable initialized wrong!");
                }
                transformed
            }
            UKF_TransitionModel::NonLinear { f } => f(&sigma_points, u),
        };
        let x_prior = self.sigma_points_prior.mul_vec_ref(&self.weights);
        let tmp_subt = self.sigma_points_prior.sub_each_col(&x_prior);
        let P_prior = tmp_subt
            .mul_mat_ref(&self.weights_diag)
            .mul_mat_ref(&tmp_subt.transpose())
            .add_mat_ref(Q);
        (x_prior, P_prior)
    }
}
