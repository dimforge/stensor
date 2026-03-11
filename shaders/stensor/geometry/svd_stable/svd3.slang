//! Slang port of nalgebra’s 3x3 SVD implementation.


module svd3;

import stensor.geometry.svd_stable.qr;
import stensor.geometry.svd_stable.symmetric_eigen;

// The SVD of a 3x3 matrix.
public struct Svd3 {
    public float3x3 U;
    public float3 S;
    public float3x3 Vt;

    public __init(float3x3 U, float3 S, float3x3 Vt) {
        this.U = U;
        this.S = S;
        this.Vt = Vt;
    }

    // Computes the SVD of a 3x3 matrix.
    public __init(float3x3 m) {
        let eps = 1.1920929e-7; // f32 epsilon

        // Compute the Gram matrix s = m^T * m.
        // Slang's mul(A, B) = A * B in row-vector convention.
        // transpose(m) * m = mul(transpose(m), m) but we need column-major [f32;9].
        let s = mul(transpose(m), m);

        // Convert float3x3 (row-major: m[row].col) to column-major array.
        // Column k = [s[0][k], s[1][k], s[2][k]].
        float s_arr[9];
        s_arr[0] = s[0].x; s_arr[1] = s[1].x; s_arr[2] = s[2].x; // col 0
        s_arr[3] = s[0].y; s_arr[4] = s[1].y; s_arr[5] = s[2].y; // col 1
        s_arr[6] = s[0].z; s_arr[7] = s[1].z; s_arr[8] = s[2].z; // col 2

        // Eigendecompose s (symmetric).
        let eigen = try_symmetric_eigen3(s_arr, eps, 0);
        var v = eigen.eigenvectors;

        // Compute b = m * v.
        // Column-major v → float3x3 (row k = [v[k], v[3+k], v[6+k]]).
        float3x3 v_mat = float3x3(
            float3(v[0], v[3], v[6]),
            float3(v[1], v[4], v[7]),
            float3(v[2], v[5], v[8])
        );
        let b_mat = mul(m, v_mat);

        // float3x3 → column-major array.
        float b[9];
        b[0] = b_mat[0].x; b[1] = b_mat[1].x; b[2] = b_mat[2].x; // col 0
        b[3] = b_mat[0].y; b[4] = b_mat[1].y; b[5] = b_mat[2].y; // col 1
        b[6] = b_mat[0].z; b[7] = b_mat[1].z; b[8] = b_mat[2].z; // col 2

        // Sort singular values in descending order.
        var rho0 = col_norm_squared(b, 0);
        var rho1 = col_norm_squared(b, 1);
        var rho2 = col_norm_squared(b, 2);

        if (rho0 < rho1) {
            swap_columns(b, 0, 1);
            neg_column(b, 1);
            swap_columns(v, 0, 1);
            neg_column(v, 1);
            let tmp = rho0; rho0 = rho1; rho1 = tmp;
        }
        if (rho0 < rho2) {
            swap_columns(b, 0, 2);
            neg_column(b, 2);
            swap_columns(v, 0, 2);
            neg_column(v, 2);
            let tmp = rho0; rho0 = rho2; rho2 = tmp;
        }
        if (rho1 < rho2) {
            swap_columns(b, 1, 2);
            neg_column(b, 2);
            swap_columns(v, 1, 2);
            neg_column(v, 2);
        }

        // QR decompose b.
        let qr_result = QR3(b);

        float q_arr[9];
        qr_result.q(q_arr);
        // Column-major q_arr → row-major float3x3: m[row] = (arr[0*3+row], arr[1*3+row], arr[2*3+row])
        this.U = float3x3(
            float3(q_arr[0], q_arr[3], q_arr[6]),
            float3(q_arr[1], q_arr[4], q_arr[7]),
            float3(q_arr[2], q_arr[5], q_arr[8])
        );

        let d = qr_result.diag_internal();
        this.S = float3(abs(d.x), abs(d.y), abs(d.z));

        // Column-major v is V. Vt(row,col) = V(col,row) = v[row*3+col].
        // So Vt[row] = float3(v[row*3], v[row*3+1], v[row*3+2]).
        this.Vt = float3x3(
            float3(v[0], v[1], v[2]),
            float3(v[3], v[4], v[5]),
            float3(v[6], v[7], v[8])
        );
    }

    // Rebuilds the matrix this svd is the decomposition of: M = U * diag(S) * Vt.
    public func recompose() -> float3x3 {
        // U * diag(S): scale column j of U by S_j → component-wise multiply each row by S.
        let U_S = float3x3(this.U[0] * this.S, this.U[1] * this.S, this.U[2] * this.S);
        return mul(U_S, this.Vt);
    }
};

func col_norm_squared(float m[9], int col) -> float {
    let base = col * 3;
    return m[base] * m[base] + m[base + 1] * m[base + 1] + m[base + 2] * m[base + 2];
}

func swap_columns(inout float m[9], int col_a, int col_b) {
    let base_a = col_a * 3;
    let base_b = col_b * 3;
    for (int k = 0; k < 3; k++) {
        let tmp = m[base_a + k];
        m[base_a + k] = m[base_b + k];
        m[base_b + k] = tmp;
    }
}

func neg_column(inout float m[9], int col) {
    let base = col * 3;
    for (int k = 0; k < 3; k++) {
        m[base + k] = -m[base + k];
    }
}
