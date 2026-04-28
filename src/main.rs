use core::time;
use rand::rng;
use rand_distr::{Distribution, Normal};
use std::{fmt::DebugList, thread};

const ACCEL_COEF: f64 = 0.1;
const SPEED_COEF: f64 = 0.4;
const POS_COEF: f64 = 0.5;

const DELTA_TIME: f64 = 5.;

fn main() {
    let mut rng = rng();
    let noise = Normal::new(0., 100.).unwrap();
    // Real values
    let mut real_position = 30_250.;
    let mut real_speed = 50.;
    let mut real_accel = 0.;
    // Next Predicted Values;
    let mut next_pre_position = 30_250.;
    let mut next_pre_speed = 50.;
    let mut next_pre_accel = 0.;
    // Current Predicted Values;
    let mut curr_pre_position = 0.;
    let mut curr_pre_speed = 0.;
    let mut curr_pre_accel = 0.;
    let sleep_time = time::Duration::from_millis(5000);
    for i in 0..11 {
        if i == 5 {
            real_accel = 8.;
        }

        let meas = real_position + noise.sample(&mut rng);
        // Current value predictions
        curr_pre_position = next_pre_position + POS_COEF * (meas - next_pre_position);
        curr_pre_speed = next_pre_speed + SPEED_COEF * ((meas - next_pre_position) / DELTA_TIME);
        curr_pre_accel =
            next_pre_accel + ACCEL_COEF * ((meas - next_pre_position) / (DELTA_TIME * DELTA_TIME));
        // Predict next values
        next_pre_position = curr_pre_position
            + (DELTA_TIME * curr_pre_speed)
            + (DELTA_TIME * DELTA_TIME * curr_pre_accel * 0.5);
        next_pre_speed = curr_pre_speed + (DELTA_TIME * curr_pre_accel);
        next_pre_accel = curr_pre_accel;
        println!(
            "({}) - Real Pos: {}, Predicted Pos: {}, Diff: {}",
            i as f64 * DELTA_TIME,
            real_position,
            curr_pre_position,
            real_position - curr_pre_position
        );
        println!(
            "({}) - Real Speed: {}, Predicted Speed: {}, Diff: {}",
            i as f64 * DELTA_TIME,
            real_speed,
            curr_pre_speed,
            real_speed - curr_pre_speed
        );
        println!(
            "({}) - Real Accel: {}, Predicted Accel: {}, Diff: {}",
            i as f64 * DELTA_TIME,
            real_accel,
            curr_pre_accel,
            real_accel - curr_pre_accel
        );
        // Update Real Values
        real_position += real_speed * DELTA_TIME + real_accel * DELTA_TIME * DELTA_TIME;
        real_speed += real_accel * DELTA_TIME;
        thread::sleep(sleep_time);
    }
}
