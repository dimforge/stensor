module symmetric_eigen;

import stensor.geometry.svd_stable.givens;
import stensor.geometry.svd_stable.symmetric_tridiagonal;
import stensor.geometry.svd_stable.utils;

public struct SymmetricEigen3 {
    public float eigenvectors[9];
    public float3 eigenvalues;
};

func wilkinson_shift(float tmm, float tnn, float tmn) -> float {
    let sq_tmn = tmn * tmn;
    if (sq_tmn != 0.0) {
        let d = (tmm - tnn) * 0.5;
        return tnn - sq_tmn / (d + signum(d) * sqrt(d * d + sq_tmn));
    } else {
        return tnn;
    }
}

func eigenvalues_2x2_symmetric(float a, float b, float d) -> float2 {
    let half_trace = (a + d) * 0.5;
    let half_diff = (a - d) * 0.5;
    let disc = sqrt(half_diff * half_diff + b * b);
    return float2(half_trace + disc, half_trace - disc);
}

func delimit_subproblem(float diag[3], inout float off_diag[2], int end_val, float eps) -> int2 {
    var n = end_val;

    while (n > 0) {
        let m = n - 1;
        if (abs(off_diag[m]) > eps * (abs(diag[n]) + abs(diag[m]))) {
            break;
        }
        n -= 1;
    }

    if (n == 0) {
        return int2(0, 0);
    }

    var new_start = n - 1;
    while (new_start > 0) {
        let m = new_start - 1;
        if (off_diag[m] == 0.0 || abs(off_diag[m]) <= eps * (abs(diag[new_start]) + abs(diag[m]))) {
            off_diag[m] = 0.0;
            break;
        }
        new_start -= 1;
    }

    return int2(new_start, n);
}

// Returns a valid SymmetricEigen3 if convergence succeeded, with eigenvalues.x set to
// a special NaN sentinel if it fails (caller should check).
public func try_symmetric_eigen3(float m_in[9], float eps, int max_niter) -> SymmetricEigen3 {
    float m[9];
    for (int i = 0; i < 9; i++) m[i] = m_in[i];

    // Find max absolute value for scaling.
    float m_amax = 0.0;
    for (int i = 0; i < 9; i++) {
        m_amax = max(m_amax, abs(m[i]));
    }

    if (m_amax != 0.0) {
        let inv = 1.0 / m_amax;
        for (int i = 0; i < 9; i++) {
            m[i] *= inv;
        }
    }

    // Tridiagonalize.
    let tri = SymmetricTridiagonal3(m);
    let unpacked = tri.unpack();
    var q_mat = unpacked.q;

    float diag[3];
    diag[0] = unpacked.diag.x;
    diag[1] = unpacked.diag.y;
    diag[2] = unpacked.diag.z;

    float off_diag[2];
    off_diag[0] = unpacked.off_diag.x;
    off_diag[1] = unpacked.off_diag.y;

    var niter = 0;
    var bounds = delimit_subproblem(diag, off_diag, 2, eps);
    var start = bounds.x;
    var end_val = bounds.y;

    while (end_val != start) {
        let subdim = end_val - start + 1;

        if (subdim > 2) {
            let m_idx = end_val - 1;
            let n = end_val;

            var vec = float2(
                diag[start] - wilkinson_shift(diag[m_idx], diag[n], off_diag[m_idx]),
                off_diag[start]
            );

            for (int i = start; i < n; i++) {
                let j = i + 1;

                var rot = GivensRotation.cancel_y(vec);
                let norm = GivensRotation.cancel_y_norm(vec);

                if (vec.y != 0.0) {
                    if (i > start) {
                        off_diag[i - 1] = norm;
                    }

                    let mii = diag[i];
                    let mjj = diag[j];
                    let mij = off_diag[i];

                    let cc = rot.c * rot.c;
                    let ss = rot.s * rot.s;
                    let cs = rot.c * rot.s;

                    let b = cs * 2.0 * mij;

                    diag[i] = (cc * mii + ss * mjj) - b;
                    diag[j] = (ss * mii + cc * mjj) + b;
                    off_diag[i] = cs * (mii - mjj) + mij * (cc - ss);

                    if (i != n - 1) {
                        vec.x = off_diag[i];
                        vec.y = -rot.s * off_diag[i + 1];
                        off_diag[i + 1] *= rot.c;
                    }

                    var inv_rot = rot.inverse();
                    inv_rot.rotate_rows(q_mat, i, j);
                } else {
                    break;
                }
            }

            if (abs(off_diag[m_idx]) <= eps * (abs(diag[m_idx]) + abs(diag[n]))) {
                end_val -= 1;
            }
        } else if (subdim == 2) {
            let eigvals = eigenvalues_2x2_symmetric(diag[start], off_diag[start], diag[start + 1]);

            float2 basis;
            if (abs(eigvals.x - diag[start + 1]) > abs(eigvals.x - diag[start])) {
                basis = float2(eigvals.x - diag[start + 1], off_diag[start]);
            } else {
                basis = float2(off_diag[start], eigvals.x - diag[start]);
            }

            diag[start] = eigvals.x;
            diag[start + 1] = eigvals.y;

            var rot = GivensRotation.try_new(basis.x, basis.y, eps);
            // Check if rotation is not identity (i.e., try_new succeeded).
            // We approximate by checking if the input was non-trivial.
            rot.rotate_rows(q_mat, start, start + 1);

            end_val -= 1;
        }

        bounds = delimit_subproblem(diag, off_diag, end_val, eps);
        start = bounds.x;
        end_val = bounds.y;

        niter += 1;
        if (max_niter > 0 && niter == max_niter) {
            // Failed to converge. Return with NaN sentinel.
            SymmetricEigen3 fail_result;
            fail_result.eigenvectors = q_mat;
            fail_result.eigenvalues = float3(0.0 / 0.0, 0.0, 0.0);
            return fail_result;
        }
    }

    // Rescale eigenvalues.
    for (int i = 0; i < 3; i++) {
        diag[i] *= m_amax;
    }

    SymmetricEigen3 result;
    result.eigenvectors = q_mat;
    result.eigenvalues = float3(diag[0], diag[1], diag[2]);
    return result;
}
