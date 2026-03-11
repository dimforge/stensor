module symmetric_tridiagonal;

import stensor.geometry.svd_stable.householder;

public struct SymmetricTridiagonal3 {
    public float tri[9];
    public float off_diagonal[2];

    public __init(float m[9]) {
        for (int i = 0; i < 9; i++) this.tri[i] = m[i];

        this.off_diagonal[0] = 0.0;
        this.off_diagonal[1] = 0.0;

        // i = 0: axis is rows [1,2] of column 0
        {
            float axis[3];
            axis[0] = this.tri[1];
            axis[1] = this.tri[2];
            axis[2] = 0.0;

            let result = reflection_axis_mut(axis, 2);
            this.off_diagonal[0] = result.first_component;

            this.tri[1] = axis[0];
            this.tri[2] = axis[1];

            if (result.not_zero) {
                let sub00 = this.tri[4];
                let sub10 = this.tri[5];
                let sub11 = this.tri[8];

                float p[2];
                p[0] = 2.0 * (sub00 * axis[0] + sub10 * axis[1]);
                p[1] = 2.0 * (sub10 * axis[0] + sub11 * axis[1]);

                let dot_val = axis[0] * p[0] + axis[1] * p[1];

                this.tri[4] = sub00 - 2.0 * p[0] * axis[0] + 2.0 * dot_val * axis[0] * axis[0];
                this.tri[5] = sub10 - p[1] * axis[0] - axis[1] * p[0] + 2.0 * dot_val * axis[1] * axis[0];
                this.tri[8] = sub11 - 2.0 * p[1] * axis[1] + 2.0 * dot_val * axis[1] * axis[1];
            }
        }

        // i = 1: axis is row [2] of column 1
        {
            float axis[3];
            axis[0] = this.tri[5];
            axis[1] = 0.0;
            axis[2] = 0.0;

            let result = reflection_axis_mut(axis, 1);
            this.off_diagonal[1] = result.first_component;

            this.tri[5] = axis[0];
        }
    }

    public func diagonal() -> float3 {
        return float3(this.tri[0], this.tri[4], this.tri[8]);
    }

    public func q(out float result[9]) {
        assemble_q(this.tri, this.off_diagonal, result);
    }

    public struct Unpacked {
        public float q[9];
        public float3 diag;
        public float2 off_diag;
    };

    public func unpack() -> Unpacked {
        Unpacked result;
        result.diag = this.diagonal();
        this.q(result.q);
        result.off_diag = float2(abs(this.off_diagonal[0]), abs(this.off_diagonal[1]));
        return result;
    }
};
