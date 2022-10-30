use nalgebra::DMatrix;

#[link(name = "ruda", kind = "static")]
extern "C" {
    pub fn ruda_mm32(
        a: *const f32,
        b: *const f32,
        m: usize,
        k: usize,
        n: usize
    ) -> *mut f32;
}

pub fn gpu_mm(lhs: &DMatrix<f32>, rhs: &DMatrix<f32>) -> DMatrix<f32> {
    unsafe {
        let elems = std::slice::from_raw_parts(
            ruda_mm32(lhs.as_ptr(), rhs.as_ptr(), lhs.nrows(), lhs.ncols(), rhs.ncols()),
            lhs.nrows() * rhs.ncols()
        );
        DMatrix::from_column_slice(lhs.nrows(), rhs.ncols(), elems)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn matmul() {
        let a = DMatrix::from_column_slice(3, 4,
            &[8.0, 2.0, 7.0, 3.0, 5.0, 6.0, 0.0, 4.0, 10.0, 1.0, 9.0, 13.0]
        );
        let b = DMatrix::from_column_slice(4, 4,
            &[5.0, 4.0, 3.0, 1.0, 8.0, 6.0, 7.0, 0.5, 0.0, 3.5, 2.4, 1.0, 6.6, 0.1, 9.5, 7.4]
        );
        let c = gpu_mm(&a, &b);

        assert_eq!(3, c.nrows());
        assert_eq!(4, c.ncols());
        assert_approx_eq!(53.0, c[(0, 0)]);
        assert_approx_eq!(78.5, c[(1, 1)]);
        assert_approx_eq!(58.0, c[(2, 2)]);
    }
}

