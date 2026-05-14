use core::f64;
use std::{
    fmt::Display,
    ops::{Add, AddAssign, SubAssign},
};

use matrix::{
    matrix::{funcs::inverse::identity_matrix, Matrix},
    vector::Vector,
};
use num_traits::{pow, Float};

enum KalmanInitState {
    /// Nothing Done Yet
    Empty,
    /// State Vector Available
    StateVecOk,
    /// State Transition Matrix Available
    StateTransMatOk,
    /// Control Matrix Available
    ControlMatOk,
    /// Estimate Covariance Matrix Available
    EstCovarianceOk,
    /// Process Noise Covariance Available
    ProcNoiseCovOk,
    /// Measurement Covariance Available
    MeasCovarianceOk,
    /// Kalman Gain Available
    ObservationMatOk,
    /// Ready To Go Without Control Matrix
    RdyToGoNoG = 0b11110110,
    /// Ready To Go With Control Matrix
    RdyToGo = 0b11111110,
}

enum StateTransationMatrix<'a, K: Float> {
    Empty,
    Linear(&'a mut Matrix<K>),
    Extended(fn(&Vector<K>) -> Matrix<K>, fn(&Vector<K>) -> Matrix<K>),
}

enum ObservationMatrix<'a, K: Float> {
    Empty,
    Linear(&'a mut Matrix<K>),
    Extended(fn(&Vector<K>) -> Vector<K>, fn(&Vector<K>) -> Matrix<K>),
}

/// Kalman Elements
/// x = State Vector             -- n_x · 1
/// z = Measurement Vector       -- n_z · 1
/// F = State Transition Matrix  -- n_x · n_x
/// u = Input Variable           -- n_u · 1
/// G = Control Matrix           -- n_x · n_u
/// P = Estimate Covariance      -- n_x · n_x
/// Q = Process Noise Covariance -- n_x · n_x
/// R = Measurement Covariance   -- n_z · n_z
/// w = Process Noise Vector     -- n_x · 1
/// v = Measurement Noise Vector -- n_z · 1
/// H = Observation Matrix       -- n_z · n_x
/// K = Kalman Gain              -- n_x · n_z
/// σ   = Random Standart Deviation
/// σ_m = Measurement Standart Deviation
/// Δt  = Delta Time
/// n_x = Number of States in State Vector
/// n_z = Number of Measured States
/// n_u = Number of Elements of the Input Variable
struct Kalman<'a, K: Float> {
    x: Vector<K>,
    x_prior: Vector<K>,
    z: Vector<K>,
    F: StateTransationMatrix<'a, K>,
    G: Matrix<K>,
    P: Matrix<K>,
    P_prior: Matrix<K>,
    Q: Matrix<K>,
    R: Matrix<K>,
    H: ObservationMatrix<'a, K>,
    K: Matrix<K>,
    I: Matrix<K>,
    control_var: bool,
    n_x: usize,
    n_z: usize,
    n_u: usize,
    // EKF Part
    state: u8,
}

impl<'a, K: Float> Kalman<'a, K> {
    pub fn new(is_control_var: bool, n_x: usize, n_z: usize, n_u: usize) -> Self {
        Kalman {
            x: Vector::empty(),
            x_prior: Vector::empty(),
            z: Vector::empty(),
            F: StateTransationMatrix::Empty,
            G: Matrix::empty(),
            P: Matrix::empty(),
            P_prior: Matrix::empty(),
            Q: Matrix::empty(),
            R: Matrix::empty(),
            H: ObservationMatrix::Empty,
            K: Matrix::empty(),
            I: identity_matrix(n_x),
            control_var: is_control_var,
            state: 1 << KalmanInitState::Empty as u8,
            n_x,
            n_z,
            n_u,
        }
    }
}

impl<'a, K: Float> Kalman<'a, K> {
    fn not_empty(&mut self) {
        self.state &= !(1u8);
    }
    pub fn init_F(&mut self, F_mat: &'a mut Matrix<K>) {
        assert_eq!(
            F_mat.size(),
            (self.n_x, self.n_x),
            "State Transition Matrix size must be (n_x, n_x)"
        );
        self.F = StateTransationMatrix::Linear(F_mat);
        self.not_empty();
        self.state |= 1 << KalmanInitState::StateTransMatOk as u8;
    }

    pub fn init_F_EKF(&mut self, f: fn(&Vector<K>) -> Matrix<K>, jac: fn(&Vector<K>) -> Matrix<K>) {
        println!("Be aware!, F(x) and Jacobian must return n_x . n_x size matrices!");
        self.F = StateTransationMatrix::Extended(f, jac);
        self.not_empty();
        self.state |= 1 << KalmanInitState::StateTransMatOk as u8;
    }

    pub fn init_P(&mut self, P_mat: Matrix<K>) {
        assert_eq!(
            P_mat.size(),
            (self.n_x, self.n_x),
            "Estimate Covariance size must be (n_x, n_x)"
        );
        self.P = P_mat.to_owned();
        self.not_empty();
        self.state |= 1 << KalmanInitState::EstCovarianceOk as u8;
    }

    pub fn init_Q(&mut self, Q_mat: Matrix<K>) {
        assert_eq!(
            Q_mat.size(),
            (self.n_x, self.n_x),
            "Process Noise Covariance size must be (n_x, n_x)"
        );
        self.Q = Q_mat.to_owned();
        self.not_empty();
        self.state |= 1 << KalmanInitState::MeasCovarianceOk as u8;
    }

    pub fn init_G(&mut self, G_mat: Matrix<K>) {
        if !self.control_var {
            println!("Control matrix can't be initialized!");
            return;
        }
        assert_eq!(
            G_mat.size(),
            (self.n_x, self.n_u),
            "Control Matrix size must be (n_x, n_u)"
        );
        self.G = G_mat.to_owned();
        self.not_empty();
        self.state |= 1 << KalmanInitState::ControlMatOk as u8;
    }

    pub fn init_R(&mut self, R_mat: Matrix<K>) {
        assert_eq!(
            R_mat.size(),
            (self.n_z, self.n_z),
            "Measurement Covariance size must be (n_z, n_z)"
        );
        self.R = R_mat.to_owned();
        self.not_empty();
        self.state |= 1 << KalmanInitState::MeasCovarianceOk as u8;
    }

    pub fn init_H(&mut self, H_mat: &'a mut Matrix<K>) {
        assert_eq!(
            H_mat.size(),
            (self.n_z, self.n_x),
            "Observation Matrix size must be (n_z, n_x)"
        );
        self.H = ObservationMatrix::Linear(H_mat);
        self.not_empty();
        self.state |= 1 << KalmanInitState::ObservationMatOk as u8;
    }

    pub fn init_H_EKF(&mut self, f: fn(&Vector<K>) -> Vector<K>, jac: fn(&Vector<K>) -> Matrix<K>) {
        println!("Be aware!, H(x) and Jacobian must return n_z . n_x size matrices!");
        self.H = ObservationMatrix::Extended(f, jac);
        self.not_empty();
        self.state |= 1 << KalmanInitState::ObservationMatOk as u8;
    }

    pub fn init_x(&mut self, x_vec: Vector<K>) {
        assert_eq!(x_vec.size(), self.n_x, "State vector wrong size!");
        self.x = x_vec.to_owned();
        self.not_empty();
        self.state |= 1 << KalmanInitState::StateVecOk as u8;
    }
}

impl<'a, K: Float> Kalman<'a, K> {
    pub fn ok_check(&self) -> bool {
        assert_ne!(self.state, 1, "Filter is empty");
        if self.control_var {
            return self.state == KalmanInitState::RdyToGo as u8;
        }
        self.state == KalmanInitState::RdyToGoNoG as u8
    }
}

impl<'a, K: Float + AddAssign + SubAssign + Display> Kalman<'a, K> {
    /// Makes Prediction For Next Step
    /// If there is no Control Matrix just pass a Vector::empty()
    pub fn predict(&mut self, u_vec: Vector<K>) {
        match &self.F {
            StateTransationMatrix::Linear(f) => {
                self.P_prior = f
                    .mul_mat_ref(&self.P)
                    .mul_mat_ref(&f.transpose())
                    .add_ref(&self.Q);
                self.x_prior = f.mul_vec_ref(&self.x);
            }

            StateTransationMatrix::Extended(f, jac) => {
                self.P_prior = jac(&self.x_prior)
                    .mul_mat_ref(&self.P)
                    .mul_mat_ref(&jac(&self.x_prior))
                    .add_ref(&self.Q);
                self.x_prior = f(&self.x_prior).mul_vec_ref(&self.x);
            }
            _ => println!("State Transition Matrix not initialized!"),
        }
        if self.control_var {
            assert_eq!(
                u_vec.size(),
                self.n_u,
                "Input vector size must be {}",
                self.n_u
            );
            self.x_prior.add_ref(&self.G.mul_vec_ref(&u_vec));
        }
    }

    pub fn update(&mut self, z_vec: Vector<K>) {
        assert_eq!(
            z_vec.size(),
            self.n_z,
            "Measurement Vector must be equal to {}!",
            self.n_z
        );
        // Update Kalman Gain
        self.update_kalman_gain();
        // Update Current State Estimation Using Prior Estimation
        match &self.H {
            ObservationMatrix::Linear(h) => {
                self.x = self
                    .K
                    .mul_vec_ref(&(z_vec.sub_ref(&h.mul_vec_ref(&self.x_prior))))
                    .add_ref(&self.x_prior);
            }
            ObservationMatrix::Extended(_f, jac) => {
                self.x = self
                    .K
                    .mul_vec_ref(&(z_vec.sub_ref(&jac(&z_vec).mul_vec_ref(&self.x_prior))))
                    .add_ref(&self.x_prior);
            }
            _ => println!("Observation Matrix not initialized!"),
        }
        // Estimate Current Estimate Uncertanity
        self.update_estimation_covariance(&z_vec);
    }

    pub fn update_kalman_gain(&mut self) {
        match &self.H {
            ObservationMatrix::Linear(h) => {
                let mut tmp: Matrix<K> = h.mul_mat_ref(&self.P_prior).mul_mat_ref(&h.transpose());
                tmp.add(&self.R);
                self.K = self
                    .P_prior
                    .mul_mat_ref(&h.transpose())
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
            ObservationMatrix::Extended(_f, jac) => {
                let mut tmp: Matrix<K> = jac(&self.x_prior)
                    .mul_mat_ref(&self.P_prior)
                    .mul_mat_ref(&jac(&self.x_prior).transpose());
                tmp.add(&self.R);
                self.K = self
                    .P_prior
                    .mul_mat_ref(&jac(&self.x_prior).transpose())
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
            _ => println!("Observation Matrix not initialized!"),
        }
    }

    pub fn update_estimation_covariance(&mut self, z_vec: &Vector<K>) {
        let joseph = self.K.mul_mat_ref(&self.R).mul_mat_ref(&self.K.transpose());
        let mut main_part: Matrix<K> = Matrix::empty();
        match &self.H {
            ObservationMatrix::Linear(h) => {
                main_part = self.I.sub_ref(&self.K.mul_mat_ref(h));
            }
            ObservationMatrix::Extended(_f, jac) => {
                main_part = self.I.sub_ref(&self.K.mul_mat_ref(&jac(&z_vec)));
            }
            _ => println!("Observation Matrix not initialized!"),
        }
        self.P = main_part
            .mul_mat_ref(&self.P_prior)
            .mul_mat_ref(&main_part.transpose())
            .add_ref(&joseph);
    }
}

fn main() {
    let mut a: Kalman<f64> = Kalman::new(true, 2, 1, 1);
    const DELTA_T: f64 = 0.25;
    let mut F_mat: Matrix<f64> = Matrix::from([[1., 0.25], [0., 1.]]);
    a.init_F(&mut F_mat);
    let delta_t_4_4 = num_traits::pow(DELTA_T, 4) / 4.;
    let delta_t_3_2 = num_traits::pow(DELTA_T, 3) / 2.;
    let delta_t_2 = num_traits::pow(DELTA_T, 2);
    let mut Q_mat: Matrix<f64> =
        Matrix::from([[delta_t_4_4, delta_t_3_2], [delta_t_3_2, delta_t_2]]);
    Q_mat.scl(pow(0.1, 2));
    a.init_Q(Q_mat);
    let R_mat: Matrix<f64> = Matrix::from([[400.]]);
    a.init_R(R_mat);
    let G_mat: Matrix<f64> = Matrix::from([[0.0313], [DELTA_T]]);
    a.init_G(G_mat);
    let mut H_mat: Matrix<f64> = Matrix::from([[1., 0.]]);
    a.init_H(&mut H_mat);
    let P_mat: Matrix<f64> = Matrix::from([[500., 0.], [0., 500.]]);
    a.init_P(P_mat);

    a.init_x(Vector::from([0., 0.]));
    if a.ok_check() {
        println!("Everything ready to go!");
    } else {
        println!("Something is not working!");
    }

    for (i, meas) in measurements.iter().enumerate() {
        if i == 0 {
            a.predict(Vector::from([0.]));
        }
        a.update(Vector::from([meas.0]));
        a.predict(Vector::from([meas.1 - 9.81]));
        println!("{} ----------------------------", i + 1);
        println!("{:?} \n\n{:.4}", meas, a.x);
        println!("{:.4}", a.P);
        println!("{:.4}", a.K);
        println!("{} ----------------------------", i + 1);
    }
}
