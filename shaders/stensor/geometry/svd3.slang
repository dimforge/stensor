// Port of the glamx SVD3 implementation.
// Uses eigendecomposition of A^T * A to compute singular values and right singular vectors,
// then derives left singular vectors with Gram-Schmidt orthogonalization.

module svd3;

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
    // Returns M = U * S * V^T with singular values sorted in descending order.
    public __init(float3x3 A) {
        // Compute A^T * A (symmetric positive semi-definite).
        // Note: mul(A, B) in slang computes A * B with row-vector convention,
        // so mul(transpose(A), A) = A^T * A.
        let ata = mul(transpose(A), A);

        // Eigendecomposition of A^T * A gives:
        // - eigenvalues = singular values squared
        // - eigenvectors = right singular vectors (columns of V)
        let eigen = symmetricEigen3(ata);

        // Eigenvalues come in ascending order, reverse to get descending.
        let eigenvalues = float3(eigen.eigenvalues.z, eigen.eigenvalues.y, eigen.eigenvalues.x);
        let V = float3x3(eigen.eigenvectors[2], eigen.eigenvectors[1], eigen.eigenvectors[0]);

        // Singular values are square roots of eigenvalues.
        let s = float3(
            sqrt(max(eigenvalues.x, 0.0)),
            sqrt(max(eigenvalues.y, 0.0)),
            sqrt(max(eigenvalues.z, 0.0))
        );

        // Compute U = A * V * S^(-1) for each column, with Gram-Schmidt orthogonalization.
        let v_col0 = V[0];
        let v_col1 = V[1];
        let v_col2 = V[2];

        // Column 0
        float3 u_col0;
        if (s.x > SVD_EPSILON) {
            u_col0 = mul(A, v_col0) / s.x;
        } else {
            u_col0 = v_col0;
        }
        u_col0 = normalize(u_col0);

        // Column 1 (orthogonalize against u_col0)
        float3 u_col1;
        if (s.y > SVD_EPSILON) {
            u_col1 = mul(A, v_col1) / s.y;
        } else {
            u_col1 = anyOrthogonal(u_col0);
        }
        u_col1 = normalize(u_col1 - u_col0 * dot(u_col1, u_col0));

        // Column 2 (orthogonalize against u_col0 and u_col1)
        float3 u_col2;
        if (s.z > SVD_EPSILON) {
            u_col2 = mul(A, v_col2) / s.z;
        } else {
            u_col2 = cross(u_col0, u_col1);
        }
        u_col2 = normalize(u_col2 - u_col0 * dot(u_col2, u_col0) - u_col1 * dot(u_col2, u_col1));

        this.U = float3x3(u_col0, u_col1, u_col2);
        this.S = s;
        this.Vt = transpose(V);
    }

    // Rebuilds the matrix this svd is the decomposition of.
    public func recompose() -> float3x3 {
        let U_S = float3x3(this.U[0] * this.S.x, this.U[1] * this.S.y, this.U[2] * this.S.z);
        return mul(this.Vt, U_S);
    }
};

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

//
// Symmetric 3x3 eigendecomposition.
// Adapted from "A Robust Eigensolver for 3x3 Symmetric Matrices" by David Eberly.
// https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
//

struct SymmetricEigen3 {
    // Eigenvalues in ascending order.
    float3 eigenvalues;
    // Eigenvectors as columns.
    float3x3 eigenvectors;
};

func symmetricEigen3(mat: float3x3) -> SymmetricEigen3 {
    let eigenvalues = eigenvalues3(mat);
    let eigenvector1 = eigenvector1(mat, eigenvalues.x);
    let eigenvector2 = eigenvector2(mat, eigenvector1, eigenvalues.y);
    let eigenvector3 = cross(eigenvector1, eigenvector2);

    SymmetricEigen3 result;
    result.eigenvalues = eigenvalues;
    result.eigenvectors = float3x3(eigenvector1, eigenvector2, eigenvector3);
    return result;
}

// Computes eigenvalues of a symmetric 3x3 matrix in ascending order.
// Reference: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
func eigenvalues3(mat: float3x3) -> float3 {
    let a00 = mat[0].x;
    let a11 = mat[1].y;
    let a22 = mat[2].z;
    let a10 = mat[0].y;
    let a20 = mat[0].z;
    let a21 = mat[1].z;

    let p1 = a10 * a10 + a20 * a20 + a21 * a21;

    if (p1 == 0.0) {
        // The matrix is diagonal. Sort eigenvalues ascending.
        var e0 = a00;
        var e1 = a11;
        var e2 = a22;
        // 3-element sorting network.
        if (e0 > e1) { let t = e0; e0 = e1; e1 = t; }
        if (e1 > e2) { let t = e1; e1 = e2; e2 = t; }
        if (e0 > e1) { let t = e0; e0 = e1; e1 = t; }
        return float3(e0, e1, e2);
    }

    let q = (a00 + a11 + a22) / 3.0;
    let p2 = (a00 - q) * (a00 - q)
           + (a11 - q) * (a11 - q)
           + (a22 - q) * (a22 - q)
           + 2.0 * p1;
    let p = sqrt(p2 / 6.0);

    let mat_b = mat - q * float3x3(
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0)
    );

    let r = (p != 0.0) ? determinant(mat_b) / (2.0 * p) : 3.402823e+38;

    let pi = 3.14159265358979323846;
    float phi;
    if (r <= -1.0) {
        phi = pi / 3.0;
    } else if (r >= 1.0) {
        phi = 0.0;
    } else {
        phi = acos(r) / 3.0;
    }

    // eigenvalues satisfy eigen3 <= eigen2 <= eigen1
    let eigen1 = q + 2.0 * p * cos(phi);
    let eigen3 = q + 2.0 * p * cos(phi + 2.0 * pi / 3.0);
    let eigen2 = 3.0 * q - eigen1 - eigen3;
    return float3(eigen3, eigen2, eigen1);
}

// Eigenvector for eigenvalue with multiplicity 1.
func eigenvector1(mat: float3x3, eigenvalue: float) -> float3 {
    let id = float3x3(
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0)
    );
    let cols = mat - float3x3(
        id[0] * eigenvalue,
        id[1] * eigenvalue,
        id[2] * eigenvalue
    );

    let c0 = cols[0];
    let c1 = cols[1];
    let c2 = cols[2];

    let c0xc1 = cross(c0, c1);
    let c0xc2 = cross(c0, c2);
    let c1xc2 = cross(c1, c2);

    let d0 = dot(c0xc1, c0xc1);
    let d1 = dot(c0xc2, c0xc2);
    let d2 = dot(c1xc2, c1xc2);

    var d_max = d0;
    var i_max = 0;
    if (d1 > d_max) { d_max = d1; i_max = 1; }
    if (d2 > d_max) { d_max = d2; i_max = 2; }

    // When all eigenvalues are equal, any unit vector is valid.
    if (d_max < 1e-20) {
        return float3(1.0, 0.0, 0.0);
    }

    if (i_max == 0) {
        return c0xc1 / sqrt(d0);
    } else if (i_max == 1) {
        return c0xc2 / sqrt(d1);
    } else {
        return c1xc2 / sqrt(d2);
    }
}

// Eigenvector for eigenvalue with potential multiplicity 2.
func eigenvector2(mat: float3x3, eigenvector1: float3, eigenvalue: float) -> float3 {
    // Compute orthonormal pair (u, v) perpendicular to eigenvector1.
    let sign = (eigenvector1.z >= 0.0) ? 1.0 : -1.0;
    let a = -1.0 / (sign + eigenvector1.z);
    let b = eigenvector1.x * eigenvector1.y * a;
    let u = float3(1.0 + sign * eigenvector1.x * eigenvector1.x * a, sign * b, -sign * eigenvector1.x);
    let v = float3(b, sign + eigenvector1.y * eigenvector1.y * a, -eigenvector1.y);

    let au = mul(mat, u);
    let av = mul(mat, v);

    var m00 = dot(u, au) - eigenvalue;
    var m01 = dot(u, av);
    var m11 = dot(v, av) - eigenvalue;

    let abs_m00 = abs(m00);
    let abs_m01 = abs(m01);
    let abs_m11 = abs(m11);

    if (abs_m00 >= abs_m11) {
        let max_abs = max(abs_m00, abs_m01);
        if (max_abs > 0.0) {
            if (abs_m00 >= abs_m01) {
                m01 /= m00;
                m00 = 1.0 / sqrt(1.0 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1.0 / sqrt(1.0 + m00 * m00);
                m00 *= m01;
            }
            return m01 * u - m00 * v;
        }
    } else {
        let max_abs = max(abs_m11, abs_m01);
        if (max_abs > 0.0) {
            if (abs_m11 >= abs_m01) {
                m01 /= m11;
                m11 = 1.0 / sqrt(1.0 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1.0 / sqrt(1.0 + m11 * m11);
                m11 *= m01;
            }
            return m11 * u - m01 * v;
        }
    }

    // M is the zero matrix, any unit-length solution suffices.
    return u;
}
