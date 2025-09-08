module svd2;

import gla.utils.trig;

// The SVD of a 2x2 matrix.
public struct Svd2 {
    public float2x2 U;
    public float2 S;
    public float2x2 Vt;

    public __init(float2x2 U, float2 S, float2x2 Vt) {
        this.U = U;
        this.S = S;
        this.Vt = Vt;
    }

    // Computes the SVD of a 2x2 matrix.
    public __init(float2x2 m) {
        let e = (m[0].x + m[1].y) * 0.5;
        let f = (m[0].x - m[1].y) * 0.5;
        let g = (m[0].y + m[1].x) * 0.5;
        let h = (m[0].y - m[1].x) * 0.5;
        let q = sqrt(e * e + h * h);
        let r = sqrt(f * f + g * g);

        // Note that the singular values are always sorted because sx >= sy
        // because q >= 0 and r >= 0.
        let sx = q + r;
        let sy = q - r;
        let sy_sign = select(sy < 0.0, -1.0, 1.0);
        let singular_values = float2(sx, sy * sy_sign);

        let a1 = stable_atan2(g, f);
        let a2 = stable_atan2(h, e);
        let theta = (a2 - a1) * 0.5;
        let phi = (a2 + a1) * 0.5;
        let st = sin(theta);
        let ct = cos(theta);
        let sp = sin(phi);
        let cp = cos(phi);

        let u = float2x2(float2(cp, sp), float2(-sp, cp));
        let v_t = float2x2(float2(ct, st * sy_sign), float2(-st, ct * sy_sign));

        this.U = u;
        this.S = singular_values;
        this.Vt = v_t;
        return Svd2(u, singular_values, v_t);
    }

    // Rebuilds the matrix this svd is the decomposition of.
    public func recompose() -> float2x2 {
        let U_S = float2x2(this.U[0] * this.S.x, this.U[1] * this.S.y);
        return mul(this.Vt, U_S);
    }
};
