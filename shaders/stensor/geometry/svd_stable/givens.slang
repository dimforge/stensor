module givens;

import stensor.geometry.svd_stable.utils;

public struct GivensRotation {
    public float c;
    public float s;

    public __init() {
        this.c = 1.0;
        this.s = 0.0;
    }

    public __init(float c, float s) {
        this.c = c;
        this.s = s;
    }

    public static func try_new(float c, float s, float eps) -> GivensRotation {
        let mod0 = abs(c);
        let sign0 = signum(c);
        let denom = sqrt(mod0 * mod0 + s * s);

        if (denom > eps) {
            let norm = sign0 * denom;
            return GivensRotation(mod0 / denom, s / norm);
        } else {
            return GivensRotation();
        }
    }

    public static func cancel_y(float2 v) -> GivensRotation {
        if (v.y != 0.0) {
            let mod0 = abs(v.x);
            let sign0 = signum(v.x);
            let denom = sqrt(mod0 * mod0 + v.y * v.y);
            let c = mod0 / denom;
            let s = -v.y / (sign0 * denom);
            return GivensRotation(c, s);
        } else {
            return GivensRotation();
        }
    }

    public static func cancel_y_norm(float2 v) -> float {
        if (v.y != 0.0) {
            let mod0 = abs(v.x);
            let sign0 = signum(v.x);
            let denom = sqrt(mod0 * mod0 + v.y * v.y);
            return sign0 * denom;
        } else {
            return 0.0;
        }
    }

    public func inverse() -> GivensRotation {
        return GivensRotation(this.c, -this.s);
    }

    // Performs rhs = self * rhs on rows (row_i, row_j) of a column-major 3x3 matrix.
    public [mutating] func rotate(inout float m[9], int row_i, int row_j) {
        let s = this.s;
        let c = this.c;

        for (int col = 0; col < 3; col++) {
            let idx_a = col * 3 + row_i;
            let idx_b = col * 3 + row_j;
            let a = m[idx_a];
            let b = m[idx_b];
            m[idx_a] = a * c - s * b;
            m[idx_b] = s * a + b * c;
        }
    }

    // Performs lhs = lhs * self on columns (col_i, col_j) of a column-major 3x3 matrix.
    public [mutating] func rotate_rows(inout float m[9], int col_i, int col_j) {
        let s = this.s;
        let c = this.c;

        for (int row = 0; row < 3; row++) {
            let idx_a = col_i * 3 + row;
            let idx_b = col_j * 3 + row;
            let a = m[idx_a];
            let b = m[idx_b];
            m[idx_a] = a * c + s * b;
            m[idx_b] = -s * a + b * c;
        }
    }
};
