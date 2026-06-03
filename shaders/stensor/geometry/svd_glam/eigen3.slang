//! Eigendecomposition of a symmetric 3x3 matrix.
//!
//! Direct Slang port of glamx's `eigen3.rs` (`SymmetricEigen3`), itself a Rust
//! adaptation of "A Robust Eigensolver for 3x3 Symmetric Matrices" by David Eberly.
//! <https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf>
//!
//! Convention: matrices are row-indexed (`m[row]` is a row), and the eigenvectors
//! are stored as the columns of `eigenvectors` (so column i pairs with eigenvalue i),
//! matching glam's column-major `Mat3::from_cols`.

module eigen3;

// The eigen decomposition of a symmetric 3x3 matrix.
public struct SymmetricEigen3 {
    // Eigenvalues, in ascending order (eigen1 <= eigen2 <= eigen3) as produced by `__init`.
    public float3 eigenvalues;
    // The three eigenvectors, stored as the columns of this matrix.
    public float3x3 eigenvectors;

    public __init(float3 eigenvalues, float3x3 eigenvectors) {
        this.eigenvalues = eigenvalues;
        this.eigenvectors = eigenvectors;
    }

    // Computes the eigen decomposition of the given symmetric 3x3 matrix.
    // Eigenvalues are returned in ascending order; use `reverse()` for descending.
    public __init(float3x3 mat) {
        let vals = eigenvalues3(mat);
        let ev1 = eigenvector1(mat, vals.x);
        let ev2 = eigenvector2(mat, ev1, vals.y);
        let ev3 = cross(ev1, ev2); // eigenvector3
        this.eigenvalues = vals;
        this.eigenvectors = matFromCols(ev1, ev2, ev3);
    }

    // Reverses the order of the eigenvalues and their corresponding eigenvectors
    // (turning ascending order into descending order).
    public func reverse() -> SymmetricEigen3 {
        let c0 = matCol(this.eigenvectors, 0);
        let c1 = matCol(this.eigenvectors, 1);
        let c2 = matCol(this.eigenvectors, 2);
        return SymmetricEigen3(
            float3(this.eigenvalues.z, this.eigenvalues.y, this.eigenvalues.x),
            matFromCols(c2, c1, c0)
        );
    }
};

// Builds a float3x3 (row-indexed) whose columns are c0, c1, c2.
public func matFromCols(c0: float3, c1: float3, c2: float3) -> float3x3 {
    return float3x3(
        float3(c0.x, c1.x, c2.x),
        float3(c0.y, c1.y, c2.y),
        float3(c0.z, c1.z, c2.z)
    );
}

// Extracts column `i` of a row-indexed float3x3.
public func matCol(m: float3x3, i: int) -> float3 {
    return float3(m[0][i], m[1][i], m[2][i]);
}

// Matrix-vector product M * v (M row-indexed, v a column vector).
func matVec(m: float3x3, v: float3) -> float3 {
    return float3(dot(m[0], v), dot(m[1], v), dot(m[2], v));
}

// Computes the eigenvalues of a symmetric 3x3 matrix, in ascending order.
// Reference: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
func eigenvalues3(mat: float3x3) -> float3 {
    // Symmetric matrix: a01 == a10, a02 == a20, a12 == a21.
    let a00 = mat[0].x;
    let a11 = mat[1].y;
    let a22 = mat[2].z;
    let a01 = mat[0].y;
    let a02 = mat[0].z;
    let a12 = mat[1].z;

    let p1 = a01 * a01 + a02 * a02 + a12 * a12;

    if (p1 == 0.0) {
        // The matrix is diagonal. Sort the diagonal entries ascending.
        var e0 = a00;
        var e1 = a11;
        var e2 = a22;
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

    let id = float3x3(
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0)
    );

    // NOTE: `p` shouldn't be zero here, but denormal flushing on some GPUs can make it so.
    var r = 1.0; // Will result in phi == 0.0.
    if (p != 0.0) {
        let mat_b = (1.0 / p) * (mat - q * id);
        r = determinant(mat_b) / 2.0;
    }

    // r should be in [-1, 1] for a symmetric matrix; clamp against round-off.
    let pi = 3.14159265358979323846;
    float phi;
    if (r <= -1.0) {
        phi = pi / 3.0;
    } else if (r >= 1.0) {
        phi = 0.0;
    } else {
        phi = acos(r) / 3.0;
    }

    // The eigenvalues satisfy eigen3 <= eigen2 <= eigen1.
    let eigen1 = q + 2.0 * p * cos(phi);
    let eigen3 = q + 2.0 * p * cos(phi + 2.0 * pi / 3.0);
    let eigen2 = 3.0 * q - eigen1 - eigen3; // trace(mat) = eigen1 + eigen2 + eigen3
    return float3(eigen3, eigen2, eigen1);
}

// Eigenvector for an eigenvalue computed from a cubic root with multiplicity 1.
func eigenvector1(mat: float3x3, eigenvalue1: float) -> float3 {
    let id = float3x3(
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0)
    );
    let cols = mat - eigenvalue1 * id;

    // `cols` is symmetric, so its rows equal its columns.
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

    // When all eigenvalues are equal (multiplicity 3), mat - eigenvalue * I is the
    // zero matrix, so all cross products vanish; any unit vector is a valid eigenvector.
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

// Eigenvector for an eigenvalue computed from a cubic root with potential multiplicity 2.
func eigenvector2(mat: float3x3, eigenvector1: float3, eigenvalue2: float) -> float3 {
    // Right-handed orthonormal set { u, v, eigenvector1 }.
    let pair = anyOrthonormalPair(eigenvector1);
    let u = pair.u;
    let v = pair.v;

    let au = matVec(mat, u);
    let av = matVec(mat, v);

    var m00 = dot(u, au) - eigenvalue2;
    var m01 = dot(u, av);
    var m11 = dot(v, av) - eigenvalue2;

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

    // M is the zero matrix; any unit-length solution suffices.
    return u;
}

struct OrthonormalPair {
    float3 u;
    float3 v;
};

// Computes any orthonormal pair of vectors perpendicular to `w` (assumed unit length).
func anyOrthonormalPair(w: float3) -> OrthonormalPair {
    let sign = (w.z >= 0.0) ? 1.0 : -1.0;
    let a = -1.0 / (sign + w.z);
    let b = w.x * w.y * a;
    OrthonormalPair result;
    result.u = float3(1.0 + sign * w.x * w.x * a, sign * b, -sign * w.x);
    result.v = float3(b, sign + w.y * w.y * a, -w.y);
    return result;
}
