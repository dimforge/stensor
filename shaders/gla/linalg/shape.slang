module shape;

public struct Shape {
    // Number of rows in each matrix of the tensor.
    public uint nrows;
    // Number of columns in each matrix of the tensor.
    public uint ncols;
    // Number of matrices in the tensor.
    public uint nmats;
    // Number of cubes (3-tensors) in the tensor.
    public uint ncubes;
    // Number of elements between two successive rows.
    public uint row_stride;
    // Number of elements between two successive columns.
    public uint col_stride;
    // Number of elements between two successive matrices.
    public uint mat_stride;
    // Number of elements between two successive cubes (3-tensors).
    public uint cube_stride;

    public func len() -> uint {
        return this.nrows * this.ncols * this.nmats * this.ncubes;
    }

    // TODO: give the same name to all these methods, slang supports overloading.
    public func iv(i: uint) -> uint {
        return i * this.row_stride;
    }

    // NOTE: this is column-major.
    public func im(i: uint, j: uint) -> uint {
        return i * this.row_stride + j * this.col_stride;
    }

    // Index of the element at row `i`, column `j` of the matrix `k` in this tensor.
    public func it(i: uint, j: uint, k: uint) -> uint {
        return i * this.row_stride + j * this.col_stride + k * this.mat_stride;
    }

    // Index of the element at row `i`, column `j` of the matrix `k` of the cube `l` in this tensor.
    public func it(i: uint, j: uint, k: uint, l: uint) -> uint {
        return i * this.row_stride + j * this.col_stride + k * this.mat_stride + l * this.cube_stride;
    }

    // Index of the element at row `i`, column `j` of the matrix `k` of the cube `l` in this tensor.
    public func it(id: uint4) -> uint {
        return id.x * this.row_stride + id.y * this.col_stride + id.z * this.mat_stride + id.w * this.cube_stride;
    }

        /// Indexes the tensor, but overflowing indices wrap around the dimension.
        ///
        /// For example if the row index is 4 and the number of rows is 3, the row index
        /// effectively used by this function is `4 % 3 = 1`.
        public func it_wrapping(i: uint, j: uint, k: uint, l: uint) -> uint {
            return this.it(i % this.nrows, j % this.ncols, k % this.nmats, l % this.ncubes);
        }

    /// Indexes the tensor, but overflowing indices wrap around the dimension.
    ///
    /// For example if the row index is 4 and the number of rows is 3, the row index
    /// effectively used by this function is `4 % 3 = 1`.
    public func it_wrapping(id: uint4) -> uint {
        return this.it(id % (uint4(this.nrows, this.ncols, this.nmats, this.ncubes)));
    }

    /// Decomposes a linear index `i` into a 4D tensor index.
    public func decompose(i: uint) -> uint4 {
        let i3 = i / (this.nrows * this.ncols * this.nmats);
        let i3_offset = i3 * (this.nmats * this.ncols * this.nrows);
        let i2 = (i - i3_offset) / (this.nrows * this.ncols);
        let i2_offset = i2 * (this.nrows * this.ncols);
        let i1 = (i - i3_offset - i2_offset) / this.nrows;
        let i0 = i - i3_offset - i2_offset - i1 * this.nrows;
        return uint4(i0, i1, i2, i3);
    }
}

public func div_ceil4(a: uint) -> uint {
    return (a + 3u) / 4u;
}