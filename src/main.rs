use core::f64;
use std::{
    fmt::Display,
    ops::{AddAssign, SubAssign},
};

use matrix::{
    matrix::{funcs::inverse::identity_matrix, Matrix},
    vector::Vector,
};
use num_traits::Float;

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
struct Kalman<K: Float> {
    x: Vector<K>,
    x_prior: Vector<K>,
    z: Vector<K>,
    F: Matrix<K>,
    u: Vector<K>,
    G: Matrix<K>,
    P: Matrix<K>,
    P_prior: Matrix<K>,
    Q: Matrix<K>,
    R: Matrix<K>,
    w: Vector<K>,
    v: Vector<K>,
    H: Matrix<K>,
    K: Matrix<K>,
    delta_time: K,
    std_dev_meas: K,
    std_dev_rand: K,
    control_var: bool,
    n_x: usize,
    n_z: usize,
    n_u: usize,
    state: u8,
}

impl<K: Float> Kalman<K> {
    pub fn new(is_control_var: bool, n_x: usize, n_z: usize, n_u: usize) -> Self {
        Kalman {
            x: Vector::empty(),
            x_prior: Vector::empty(),
            z: Vector::empty(),
            F: Matrix::empty(),
            u: Vector::empty(),
            G: Matrix::empty(),
            P: Matrix::empty(),
            P_prior: Matrix::empty(),
            Q: Matrix::empty(),
            R: Matrix::empty(),
            w: Vector::empty(),
            v: Vector::empty(),
            H: Matrix::empty(),
            K: Matrix::empty(),
            delta_time: K::zero(),
            std_dev_meas: K::zero(),
            std_dev_rand: K::zero(),
            control_var: is_control_var,
            state: KalmanInitState::Empty as u8,
            n_x,
            n_z,
            n_u,
        }
    }
}

impl<K: Float> Kalman<K> {
    fn not_empty(&mut self) {
        self.state &= !(1u8);
    }
    pub fn init_F(&mut self, F_mat: Matrix<K>) {
        assert_eq!(
            F_mat.size(),
            (self.n_x, self.n_x),
            "State Transition Matrix size must be (n_x, n_x)"
        );
        self.F = F_mat.to_owned();
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
            "Estimate Covariance size must be (n_x, n_x)"
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
            (self.n_x, self.n_x),
            "Estimate Covariance size must be (n_x, n_x)"
        );
        self.G = G_mat.to_owned();
        self.not_empty();
        self.state |= 1 << KalmanInitState::ControlMatOk as u8;
    }

    pub fn init_R(&mut self, R_mat: Matrix<K>) {
        assert_eq!(
            R_mat.size(),
            (self.n_z, self.n_z),
            "Estimate Covariance size must be (n_z, n_z)"
        );
        self.R = R_mat.to_owned();
        self.not_empty();
        self.state |= 1 << KalmanInitState::MeasCovarianceOk as u8;
    }

    pub fn init_H(&mut self, H_mat: Matrix<K>) {
        assert_eq!(
            H_mat.size(),
            (self.n_z, self.n_x),
            "Estimate Covariance size must be (n_z, n_x)"
        );
        self.H = H_mat.to_owned();
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

impl<K: Float> Kalman<K> {
    pub fn ok_check(&self) -> bool {
        assert_ne!(self.state, 1, "Filter is empty");
        if self.control_var {
            return self.state == KalmanInitState::RdyToGo as u8;
        }
        self.state == KalmanInitState::RdyToGoNoG as u8
    }
}

impl<K: Float + AddAssign + SubAssign + Display> Kalman<K> {
    /// Makes Prediction For Next Step (Control Variable not Available
    pub fn predict(&mut self, u_vec: Vector<K>) {
        self.P_prior = self.F.mul_mat_ref(&self.P).mul_mat_ref(&self.F.transpose());
        self.x_prior = self.F.mul_vec_ref(&self.x);
        if self.control_var {
            assert_eq!(
                u_vec.size(),
                self.n_u,
                "Input vector size must be {}",
                self.n_u
            );
            self.x_prior.add(&self.G.mul_vec_ref(&self.u));
        }
        println!("Prediction Result");
        println!("P_prior: {}", self.P_prior);
        println!("x_prior: {}", self.x_prior);
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
        println!("Starting To Estimate \n{}", z_vec);
        println!("Sizes \n{}, \n{}", self.H, self.x_prior);
        println!("Size \n{}", self.H.mul_vec_ref(&self.x_prior));
        let tmp = z_vec - self.H.mul_vec_ref(&self.x_prior);
        self.x = self.x_prior.clone() + tmp;
        // Estimate Current Estimate Uncertanity
        self.update_estimation_covariance();
    }

    pub fn update_kalman_gain(&mut self) {
        let mut tmp: Matrix<K> = self
            .H
            .mul_mat_ref(&self.P_prior)
            .mul_mat_ref(&self.H.transpose());
        tmp.add(&self.R);
        self.K = self
            .P_prior
            .mul_mat_ref(&self.H.transpose())
            .mul_mat_ref(&tmp.inverse().unwrap());
        println!("New kalman gain \n{}", self.K);
    }

    pub fn update_estimation_covariance(&mut self) {
        println!("Error in this line!");
        let joseph = self.K.mul_mat_ref(&self.R).mul_mat_ref(&self.K.transpose());
        let mut main_part: Matrix<K> = identity_matrix(self.n_x) - self.K.mul_mat_ref(&self.H);
        self.P = main_part
            .mul_mat_ref(&self.P_prior)
            .mul_mat_ref(&main_part.transpose())
            + joseph;
    }
}

fn main() {
    let measurements = [
        (301.5, -401.46),
        (298.23, -375.44),
        (297.83, -346.15),
        (300.42, -320.2),
        (301.94, -300.08),
        (299.5, -274.12),
        (305.98, -253.45),
        (301.25, -226.4),
        (299.73, -200.65),
        (299.2, -171.62),
        (298.62, -152.11),
        (301.84, -125.19),
        (299.6, -93.4),
        (295.3, -74.79),
        (299.3, -49.12),
        (301.95, -28.73),
        (296.3, 2.99),
        (295.11, 25.65),
        (295.12, 49.86),
        (289.9, 72.87),
        (283.51, 96.34),
        (276.42, 120.4),
        (264.22, 144.69),
        (250.25, 168.06),
        (236.66, 184.99),
        (217.47, 205.11),
        (199.75, 221.82),
        (179.7, 238.3),
        (160., 253.02),
        (140.92, 267.19),
        (113.53, 270.71),
        (93.68, 285.86),
        (69.71, 288.48),
        (45.93, 292.9),
        (20.87, 298.77),
    ];
    let mut a: Kalman<f64> = Kalman::new(false, 6, 2, 0);
    let F_mat: Matrix<f64> = Matrix::from([
        [1., 1., 0.5, 0., 0., 0.],
        [0., 1., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0.5],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.],
    ]);
    a.init_F(F_mat);
    let Q_mat: Matrix<f64> = Matrix::from([
        [0.01, 0.02, 0.02, 0., 0., 0.],
        [0.02, 0.04, 0.04, 0., 0., 0.],
        [0.02, 0.04, 0.04, 0., 0., 0.],
        [0., 0., 0., 0.01, 0.02, 0.02],
        [0., 0., 0., 0.02, 0.04, 0.04],
        [0., 0., 0., 0.02, 0.04, 0.04],
    ]);
    a.init_Q(Q_mat);
    let R_mat: Matrix<f64> = Matrix::from([[9., 0.], [0., 9.]]);
    a.init_R(R_mat);
    let H_mat: Matrix<f64> = Matrix::from([[1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]]);
    a.init_H(H_mat);
    a.init_x(Vector::from([0., 0., 0., 0., 0., 0.]));
    let P_mat: Matrix<f64> = Matrix::from([
        [500., 0., 0., 0., 0., 0.],
        [0., 500., 0., 0., 0., 0.],
        [0., 0., 500., 0., 0., 0.],
        [0., 0., 0., 500., 0., 0.],
        [0., 0., 0., 0., 500., 0.],
        [0., 0., 0., 0., 0., 500.],
    ]);
    a.init_P(P_mat);
    if a.ok_check() {
        println!("Everything ready to go!");
    } else {
        println!("Something is not working!");
    }
    for meas in measurements {
        a.predict(Vector::empty());
        a.update(Vector::from([meas.0, meas.1]));
    }
}
