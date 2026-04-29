fn calculate_kalman_gain(meas_var: f64, esti_var: f64) -> f64 {
    esti_var / (meas_var + esti_var)
}

fn main() {
    // Real and measurement values
    let real_values: [f64; 10] = [
        50.005, 49.994, 49.993, 50.001, 50.006, 49.998, 50.021, 50.005, 50., 49.997,
    ];
    let measurements: [f64; 10] = [
        49.986, 49.963, 50.09, 50.001, 50.018, 50.05, 49.938, 49.858, 49.965, 50.114,
    ];
    let q_noise = 0.0001;
    let mut next_pre_temp = 60.;
    let mut next_pre_variance = 10000. + q_noise;
    // Current Predicted Values;
    let mut curr_pre_temp = 0.;
    let mut curr_pre_variance = 0.;
    for (meas, real_val) in measurements.iter().zip(real_values) {
        let kalman_gain = calculate_kalman_gain(0.01, next_pre_variance);
        curr_pre_temp = next_pre_temp + kalman_gain * (meas - next_pre_temp);
        curr_pre_variance = (1. - kalman_gain) * next_pre_variance;
        next_pre_temp = curr_pre_temp;
        next_pre_variance = curr_pre_variance + q_noise;
        println!(
            "Real t:{}, Predicted t: {}, Meas. t: {}",
            real_val, curr_pre_temp, meas
        );
        println!(
            "Current Variance: {}, Next Variance {}",
            curr_pre_variance, next_pre_variance
        );
    }
}
