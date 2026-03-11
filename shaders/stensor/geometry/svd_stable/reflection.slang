module reflection;

public func reflect_with_sign(
    float axis[3],
    inout float m[9],
    int row_start,
    int col_start,
    int nrows,
    int ncols,
    float sign_val,
) {
    let m_two = sign_val * (-2.0);

    for (int j = 0; j < ncols; j++) {
        let col = col_start + j;
        float dot_val = 0.0;
        for (int k = 0; k < nrows; k++) {
            dot_val += axis[k] * m[col * 3 + row_start + k];
        }

        let factor = dot_val * m_two;

        for (int k = 0; k < nrows; k++) {
            let idx = col * 3 + row_start + k;
            m[idx] = sign_val * m[idx] + factor * axis[k];
        }
    }
}

public func reflect_rows_with_sign(
    float axis[3],
    inout float m[9],
    int row_start,
    int col_start,
    int nrows,
    int ncols,
    inout float work[3],
    float sign_val,
) {
    for (int i = 0; i < nrows; i++) {
        float sum = 0.0;
        for (int k = 0; k < ncols; k++) {
            let col = col_start + k;
            sum += m[col * 3 + row_start + i] * axis[k];
        }
        work[i] = sum;
    }

    let m_two = sign_val * (-2.0);
    for (int j = 0; j < ncols; j++) {
        let col = col_start + j;
        for (int i = 0; i < nrows; i++) {
            let idx = col * 3 + row_start + i;
            m[idx] = sign_val * m[idx] + m_two * work[i] * axis[j];
        }
    }
}
