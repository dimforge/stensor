module householder;

import stensor.geometry.svd_stable.reflection;
import stensor.geometry.svd_stable.utils;

public struct ReflectionAxisResult {
    public float first_component;
    public bool not_zero;
};

public func reflection_axis_mut(inout float column[3], int len) -> ReflectionAxisResult {
    float reflection_sq_norm = 0.0;
    for (int i = 0; i < len; i++) {
        reflection_sq_norm += column[i] * column[i];
    }
    let reflection_norm = sqrt(reflection_sq_norm);

    let modulus = abs(column[0]);
    let sign_val = signum(column[0]);
    let signed_norm = sign_val * reflection_norm;
    let factor = (reflection_sq_norm + modulus * reflection_norm) * 2.0;
    column[0] += signed_norm;

    if (factor != 0.0) {
        let inv_sqrt_factor = 1.0 / sqrt(factor);
        for (int i = 0; i < len; i++) {
            column[i] *= inv_sqrt_factor;
        }

        // Normalize again for numerical stability.
        float norm_sq = 0.0;
        for (int i = 0; i < len; i++) {
            norm_sq += column[i] * column[i];
        }
        let norm = sqrt(norm_sq);
        if (norm != 0.0) {
            let inv_norm = 1.0 / norm;
            for (int i = 0; i < len; i++) {
                column[i] *= inv_norm;
            }
        }

        ReflectionAxisResult result;
        result.first_component = -signed_norm;
        result.not_zero = true;
        return result;
    } else {
        ReflectionAxisResult result;
        result.first_component = signed_norm;
        result.not_zero = false;
        return result;
    }
}

public func clear_column_unchecked(inout float matrix[9], int icol, int shift) -> float {
    let start = icol + shift;
    let axis_len = 3 - start;

    float axis[3];
    axis[0] = 0.0; axis[1] = 0.0; axis[2] = 0.0;
    for (int k = 0; k < axis_len; k++) {
        axis[k] = matrix[icol * 3 + start + k];
    }

    let result = reflection_axis_mut(axis, axis_len);

    for (int k = 0; k < axis_len; k++) {
        matrix[icol * 3 + start + k] = axis[k];
    }

    if (result.not_zero) {
        let sign_val = signum(result.first_component);
        let ncols = 3 - (icol + 1);
        if (ncols > 0) {
            reflect_with_sign(
                axis,
                matrix,
                start,
                icol + 1,
                axis_len,
                ncols,
                sign_val,
            );
        }
    }

    return result.first_component;
}

public func assemble_q(float m[9], float signs[2], out float result[9]) {
    // Start with identity.
    for (int i = 0; i < 9; i++) result[i] = 0.0;
    result[0] = 1.0;
    result[4] = 1.0;
    result[8] = 1.0;

    for (int i = 1; i >= 0; i--) {
        let axis_len = 3 - (i + 1);
        float axis[3];
        axis[0] = 0.0; axis[1] = 0.0; axis[2] = 0.0;
        for (int k = 0; k < axis_len; k++) {
            axis[k] = m[i * 3 + (i + 1) + k];
        }

        let sign_val = signum(signs[i]);
        let nrows = axis_len;
        let ncols = 3 - i;
        reflect_with_sign(
            axis,
            result,
            i + 1,
            i,
            nrows,
            ncols,
            sign_val,
        );
    }
}
