//! SVD of a 3x3 matrix.
//!
//! Direct Slang port of glamx's `svd3.rs`: eigendecompose A^T * A to get the singular
//! values and right singular vectors, then derive the left singular vectors via
//! Gram-Schmidt orthogonalization.
//!
//! Public interface (struct fields, constructors and `recompose`) matches the other
//! `Svd3` implementations (`stensor.geometry.svd3`, `stensor.geometry.svd_stable.svd3`)
//! so this is a drop-in replacement: M = U * diag(S) * Vt with S in descending order.
//!
//! Accuracy notes (inherited from glamx):
//!  - Forming A^T * A squares the condition number, so the smallest singular value and the
//!    reconstruction lose accuracy on ill-conditioned / near-singular matrices.
//!  - The eigensolver uses an absolute degeneracy threshold (1e-20 on A^T*A), so inputs of very
//!    small magnitude (roughly |A| << 1) fall outside its valid domain. Inputs near unit scale
//!    (e.g. deformation gradients ~O(1)) are well within range.
//! For higher accuracy on arbitrary inputs, prefer `stensor.geometry.svd_stable.svd3`.

module svd3;

import stensor.geometry.svd_glam.eigen3;

static const float SVD_EPSILON = 1e-10;

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
    // Returns M = U * diag(S) * V^T with singular values sorted in descending order.
    public __init(float3x3 A) {
        // A^T * A is symmetric positive semi-definite. Slang's mul(X, Y) = X * Y, so
        // mul(transpose(A), A) = A^T * A.
        let ata = mul(transpose(A), A);

        // Eigendecomposition of A^T * A:
        //  - eigenvalues  = squared singular values
        //  - eigenvectors = right singular vectors (columns of V)
        // `reverse()` turns the ascending eigenvalues into descending order.
        let eigen = SymmetricEigen3(ata).reverse();

        let s = float3(
            sqrt(max(eigen.eigenvalues.x, 0.0)),
            sqrt(max(eigen.eigenvalues.y, 0.0)),
            sqrt(max(eigen.eigenvalues.z, 0.0))
        );

        // V has the right singular vectors as its columns.
        let v_col0 = matCol(eigen.eigenvectors, 0);
        let v_col1 = matCol(eigen.eigenvectors, 1);
        let v_col2 = matCol(eigen.eigenvectors, 2);

        // U.col(i) = A * V.col(i) / s[i], with Gram-Schmidt orthogonalization and
        // fallbacks for vanishing singular values.
        float3 u_col0;
        if (s.x > SVD_EPSILON) {
            u_col0 = matVec3(A, v_col0) / s.x;
        } else {
            u_col0 = v_col0;
        }
        u_col0 = normalize(u_col0);

        float3 u_col1;
        if (s.y > SVD_EPSILON) {
            u_col1 = matVec3(A, v_col1) / s.y;
        } else {
            u_col1 = anyOrthogonal(u_col0);
        }
        // Orthogonalize against u_col0.
        u_col1 = normalize(u_col1 - u_col0 * dot(u_col1, u_col0));

        float3 u_col2;
        if (s.z > SVD_EPSILON) {
            u_col2 = matVec3(A, v_col2) / s.z;
        } else {
            u_col2 = cross(u_col0, u_col1);
        }
        // Orthogonalize against u_col0 and u_col1.
        u_col2 = normalize(u_col2 - u_col0 * dot(u_col2, u_col0) - u_col1 * dot(u_col2, u_col1));

        this.U = matFromCols(u_col0, u_col1, u_col2);
        this.S = s;
        this.Vt = transpose(eigen.eigenvectors); // V^T
    }

    // Rebuilds the matrix this svd is the decomposition of: M = U * diag(S) * Vt.
    public func recompose() -> float3x3 {
        // Scale column j of U by S_j (U[row] is a row, so this is component-wise per row).
        let U_S = float3x3(this.U[0] * this.S, this.U[1] * this.S, this.U[2] * this.S);
        return mul(U_S, this.Vt);
    }
};

// Matrix-vector product M * v (M row-indexed, v a column vector).
func matVec3(m: float3x3, v: float3) -> float3 {
    return float3(dot(m[0], v), dot(m[1], v), dot(m[2], v));
}

// Find any unit vector orthogonal to the given vector.
func anyOrthogonal(v: float3) -> float3 {
    let abs_v = abs(v);
    float3 other;
    if (abs_v.x <= abs_v.y && abs_v.x <= abs_v.z) {
        other = float3(1.0, 0.0, 0.0);
    } else if (abs_v.y <= abs_v.z) {
        other = float3(0.0, 1.0, 0.0);
    } else {
        other = float3(0.0, 0.0, 1.0);
    }
    return normalize(cross(v, other));
}
