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
        self.state |= 1 << KalmanInitState::ProcNoiseCovOk as u8;
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

            StateTransationMatrix::Extended(_f, jac) => {
                self.P_prior = jac(&self.x_prior)
                    .mul_mat_ref(&self.P)
                    .mul_mat_ref(&jac(&self.x_prior))
                    .add_ref(&self.Q);
                self.x_prior = jac(&self.x).mul_vec_ref(&self.x);
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
            ObservationMatrix::Extended(f, _jac) => {
                println!("f: {}", &f(&self.x_prior).size());
                println!("K: {:?}", self.K.size());
                println!("z_vec: {:?}", z_vec.size());
                println!("x_prior: {:?}", self.x_prior.size());
                self.x = self
                    .K
                    .mul_vec_ref(&(z_vec.sub_ref(&f(&self.x_prior))))
                    .add_ref(&self.x_prior);
            }
            _ => println!("Observation Matrix not initialized!"),
        }
        // Estimate Current Estimate Uncertanity
        self.update_estimation_covariance();
    }

    pub fn update_kalman_gain(&mut self) {
        match &self.H {
            ObservationMatrix::Linear(h) => {
                let mut tmp: Matrix<K> = h.mul_mat_ref(&self.P_prior).mul_mat_ref(&h.transpose());
                tmp.add_ref(&self.R);
                self.K = self
                    .P_prior
                    .mul_mat_ref(&h.transpose())
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
            ObservationMatrix::Extended(_f, jac) => {
                let mut tmp: Matrix<K> = jac(&self.x_prior)
                    .mul_mat_ref(&self.P_prior)
                    .mul_mat_ref(&jac(&self.x_prior).transpose());
                tmp.add_ref(&self.R);
                self.K = self
                    .P_prior
                    .mul_mat_ref(&jac(&self.x_prior).transpose())
                    .mul_mat_ref(&tmp.inverse().unwrap())
            }
            _ => println!("Observation Matrix not initialized!"),
        }
    }

    pub fn update_estimation_covariance(&mut self) {
        let joseph = self.K.mul_mat_ref(&self.R).mul_mat_ref(&self.K.transpose());
        let mut main_part: Matrix<K> = Matrix::empty();
        match &self.H {
            ObservationMatrix::Linear(h) => {
                main_part = self.I.sub_ref(&self.K.mul_mat_ref(h));
            }
            ObservationMatrix::Extended(_f, jac) => {
                main_part = self.I.sub_ref(&self.K.mul_mat_ref(&jac(&self.x_prior)));
            }
            _ => println!("Observation Matrix not initialized!"),
        }
        self.P = main_part
            .mul_mat_ref(&self.P_prior)
            .mul_mat_ref(&main_part.transpose())
            .add_ref(&joseph);
    }
}

fn ekf_H_f<K: Float>(vec: &Vector<K>) -> Vector<K> {
    Vector::from([
        (vec.data[0] * vec.data[0] + vec.data[1] * vec.data[1]).sqrt(),
        (vec.data[1] / vec.data[0]).atan(),
    ])
}

fn ekf_H_jac<K: Float>(vec: &Vector<K>) -> Matrix<K> {
    Matrix::from([
        [
            vec.data[0] / (vec.data[0] * vec.data[0] + vec.data[1] * vec.data[1]).sqrt(),
            K::zero(),
            K::zero(),
            vec.data[1] / (vec.data[0] * vec.data[0] + vec.data[1] * vec.data[1]).sqrt(),
            K::zero(),
            K::zero(),
        ],
        [
            -vec.data[1] / (vec.data[0] * vec.data[0] + vec.data[1] * vec.data[1]).sqrt(),
            K::zero(),
            K::zero(),
            vec.data[0] / (vec.data[0] * vec.data[0] + vec.data[1] * vec.data[1]).sqrt(),
            K::zero(),
            K::zero(),
        ],
    ])
}

fn main() {
    let measurements = [
        (502.55, -0.9316),
        (477.34, -0.8977),
        (457.21, -0.8512),
        (442.94, -0.8114),
        (427.27, -0.7852),
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
        (300.3, 0.5934),
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
    let mut a: Kalman<f64> = Kalman::new(false, 6, 2, 0);
    let mut F_mat: Matrix<f64> = Matrix::from([
        [1., 1., 0.5, 0., 0., 0.],
        [0., 1., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0.5],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.],
    ]);
    a.init_F(&mut F_mat);
    let mut Q_mat: Matrix<f64> = Matrix::from([
        [0.25, 0.5, 0.5, 0., 0., 0.],
        [0.5, 1., 1., 0., 0., 0.],
        [0.5, 1., 1., 0., 0., 0.],
        [0., 0., 0., 0.25, 0.5, 0.5],
        [0., 0., 0., 0.5, 1., 1.],
        [0., 0., 0., 0.5, 1., 1.],
    ]);
    Q_mat.scl(pow(0.2, 2));
    a.init_Q(Q_mat);
    let R_mat: Matrix<f64> = Matrix::from([[25., 0.], [0., pow(0.0087, 2)]]);
    a.init_R(R_mat);
    a.init_H_EKF(ekf_H_f, ekf_H_jac);
    let P_mat: Matrix<f64> = Matrix::from([
        [500., 0., 0., 0., 0., 0.],
        [0., 500., 0., 0., 0., 0.],
        [0., 0., 500., 0., 0., 0.],
        [0., 0., 0., 500., 0., 0.],
        [0., 0., 0., 0., 500., 0.],
        [0., 0., 0., 0., 0., 500.],
    ]);
    a.init_P(P_mat);

    a.init_x(Vector::from([400., 0., 0., -300., 0., 0.]));
    if a.ok_check() {
        println!("Everything ready to go!");
    } else {
        println!("Something is not working!");
    }

    for (i, meas) in measurements.iter().enumerate() {
        if i == 0 {
            a.predict(Vector::empty());
        }
        a.update(Vector::from([meas.0, meas.1]));
        a.predict(Vector::empty());
        println!("{} ----------------------------", i + 1);
        println!("{:?} \n\n{:.4}", meas, a.x);
        println!("{:.4}", a.P);
        println!("{:.4}", a.K);
        println!("{} ----------------------------", i + 1);
    }
}
