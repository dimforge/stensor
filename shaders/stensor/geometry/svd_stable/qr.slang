module qr;

import stensor.geometry.svd_stable.householder;
import stensor.geometry.svd_stable.reflection;
import stensor.geometry.svd_stable.utils;

public struct QR3 {
    public float qr[9];
    public float3 diag;

    public __init(float matrix[9]) {
        for (int i = 0; i < 9; i++) this.qr[i] = matrix[i];

        float diag_arr[3];
        for (int i = 0; i < 3; i++) {
            diag_arr[i] = clear_column_unchecked(this.qr, i, 0);
        }

        this.diag = float3(diag_arr[0], diag_arr[1], diag_arr[2]);
    }

    public func q(out float result[9]) {
        for (int i = 0; i < 9; i++) result[i] = 0.0;
        result[0] = 1.0;
        result[4] = 1.0;
        result[8] = 1.0;

        float diag_arr[3];
        diag_arr[0] = this.diag.x;
        diag_arr[1] = this.diag.y;
        diag_arr[2] = this.diag.z;

        for (int i = 2; i >= 0; i--) {
            let axis_len = 3 - i;
            float axis[3];
            axis[0] = 0.0; axis[1] = 0.0; axis[2] = 0.0;
            for (int k = 0; k < axis_len; k++) {
                axis[k] = this.qr[i * 3 + i + k];
            }

            let sign_val = signum(diag_arr[i]);
            let nrows = axis_len;
            let ncols = 3 - i;
            reflect_with_sign(
                axis,
                result,
                i,
                i,
                nrows,
                ncols,
                sign_val,
            );
        }
    }

    public func diag_internal() -> float3 {
        return this.diag;
    }
};
