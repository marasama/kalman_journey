fn calculate_kalman_gain(meas_var: f64, esti_var: f64) -> f64 {
    esti_var / (meas_var + esti_var)
}

fn main() {
    // Real values
    let measurements: [f64; 10] = [
        49.03, 48.44, 55.21, 49.98, 50.6, 52.61, 45.87, 42.64, 48.26, 55.84,
    ];
    let mut next_pre_height = 60.;
    let mut next_pre_variance = 225.;
    // Current Predicted Values;
    let mut curr_pre_height = 0.;
    let mut curr_pre_variance = 0.;
    for meas in measurements {
        let kalman_gain = calculate_kalman_gain(25., next_pre_variance);
        curr_pre_height = next_pre_height + kalman_gain * (meas - next_pre_height);
        curr_pre_variance = (1. - kalman_gain) * next_pre_variance;
        next_pre_height = curr_pre_height;
        next_pre_variance = curr_pre_variance;
        println!(
            "Real h:{}, Predicted h: {}, Meas. h: {}",
            50., curr_pre_height, meas
        );
    }
}
