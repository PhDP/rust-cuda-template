use nalgebra::DMatrix;
use ndarray::{Array2, ArrayView, ShapeBuilder};

#[link(name = "ruda", kind = "static")]
extern "C" {
    /// Matrix multiplication on the GPU using cuBlas.
    /// 
    /// # Arguments
    ///
    /// * `a` - Pointer to a column-major m-by-k matrix.
    /// * `b` - Pointer to a column-major k-by-n matrix.
    /// * `m` - Number of rows of the first matrix.
    /// * `k` - Number of columns of the first matrix and rows of the second one.
    /// * `n` - Number of columns of the second matrix.
    pub fn ruda_mm32(
        a: *const f32,
        b: *const f32,
        m: libc::c_int,
        k: libc::c_int,
        n: libc::c_int
    ) -> *mut f32;
}

// Multiply two matrices using the CUDA kernel. Like CUDA, nalgebra's
// matrices adopt a column-major order by default.
pub fn nalgebra_gpu_mm(lhs: &DMatrix<f32>, rhs: &DMatrix<f32>) -> DMatrix<f32> {
    unsafe {
        DMatrix::from_column_slice(
            lhs.nrows(),
            rhs.ncols(),
            std::slice::from_raw_parts(
                ruda_mm32(
                    lhs.as_ptr(),
                    rhs.as_ptr(),
                    lhs.nrows() as libc::c_int,
                    lhs.ncols() as libc::c_int,
                    rhs.ncols() as libc::c_int
                ),
                lhs.nrows() * rhs.ncols()
            )
        )
    }
}

// Multiply two matrices using the CUDA kernel. Unlike CUDA, ndarray's
// matrices adopt a row-major order by default.
pub fn ndarray_gpu_mm(lhs: &Array2<f32>, rhs: &Array2<f32>) -> Array2<f32> {
    unsafe {
        let m = lhs.nrows();
        let k = lhs.ncols();
        let n = rhs.ncols();
        ArrayView::from_shape_ptr(
            (m, n).f(),
            ruda_mm32(
                lhs.as_ptr(),
                rhs.as_ptr(),
                m as libc::c_int,
                k as libc::c_int,
                n as libc::c_int
            )
        ).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn nalgebra_matmul() {
        let a = DMatrix::from_column_slice(3, 4,
            &[8.0, 2.0, 7.0, 3.0, 5.0, 6.0, 0.0, 4.0, 10.0, 1.0, 9.0, 13.0]
        );
        let b = DMatrix::from_column_slice(4, 4,
            &[5.0, 4.0, 3.0, 1.0, 8.0, 6.0, 7.0, 0.5, 0.0, 3.5, 2.4, 1.0, 6.6, 0.1, 9.5, 7.4]
        );
        let c = nalgebra_gpu_mm(&a, &b);

        assert_eq!(3, c.nrows());
        assert_eq!(4, c.ncols());
        assert_approx_eq!(53.0, c[(0, 0)]);
        assert_approx_eq!(78.5, c[(1, 1)]);
        assert_approx_eq!(58.0, c[(2, 2)]);
    }
    
    #[test]
    fn ndarray_matmul() {
        let a: Array2<f32> = ArrayView::from_shape((3, 4).f(),
            &[8.0, 2.0, 7.0, 3.0, 5.0, 6.0, 0.0, 4.0, 10.0, 1.0, 9.0, 13.0]
        ).unwrap().to_owned();
        let b: Array2<f32> = ArrayView::from_shape((4, 4).f(),
            &[5.0, 4.0, 3.0, 1.0, 8.0, 6.0, 7.0, 0.5, 0.0, 3.5, 2.4, 1.0, 6.6, 0.1, 9.5, 7.4]
        ).unwrap().to_owned();
        let c = ndarray_gpu_mm(&a, &b);

        assert_eq!(3, c.nrows());
        assert_eq!(4, c.ncols());
        assert_approx_eq!(53.0, c[(0, 0)]);
        assert_approx_eq!(78.5, c[(1, 1)]);
        assert_approx_eq!(58.0, c[(2, 2)]);
    }
}

