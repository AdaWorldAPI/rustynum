// NOTE: std::ops traits (Add, Sub, Mul, Div) cannot return Result, so panics
// remain in trait impls for shape mismatches — this matches NumPy's behavior.
// For fallible alternatives, use the try_* methods on NumArray (e.g. try_div_broadcast).
use super::NumArray;
use crate::simd_ops::SimdOps;
use crate::traits::{ExpLog, FromU32, FromUsize, NumOps};
use std::fmt::Debug;

use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

impl<T, Ops> Add<T> for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::add_scalar(data, rhs, &mut result_data);
        Self::new(result_data)
    }
}

impl<T, Ops> Add<T> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn add(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::add_scalar(data, rhs, &mut result_data);
        NumArray::new(result_data)
    }
}

impl<T, Ops> Add for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::add_array(data, rhs.get_data(), &mut result_data);
        Self::new(result_data)
    }
}

impl<'b, T, Ops> Add<&'b NumArray<T, Ops>> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn add(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::add_array(data, rhs.get_data(), &mut result_data);
        NumArray::new(result_data)
    }
}

impl<T, Ops> Sub<T> for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::sub_scalar(data, rhs, &mut result_data);
        Self::new(result_data)
    }
}

impl<T, Ops> Sub<T> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn sub(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::sub_scalar(data, rhs, &mut result_data);
        NumArray::new(result_data)
    }
}

impl<T, Ops> Sub for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::sub_array(data, rhs.get_data(), &mut result_data);
        Self::new(result_data)
    }
}

impl<'b, T, Ops> Sub<&'b NumArray<T, Ops>> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn sub(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::sub_array(data, rhs.get_data(), &mut result_data);
        NumArray::new(result_data)
    }
}

impl<T, Ops> Mul<T> for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::mul_scalar(data, rhs, &mut result_data);
        Self::new(result_data)
    }
}

impl<T, Ops> Mul<T> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn mul(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::mul_scalar(data, rhs, &mut result_data);
        NumArray::new(result_data)
    }
}

impl<T, Ops> Mul for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::mul_array(data, rhs.get_data(), &mut result_data);
        Self::new(result_data)
    }
}

impl<'b, T, Ops> Mul<&'b NumArray<T, Ops>> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn mul(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::mul_array(data, rhs.get_data(), &mut result_data);
        NumArray::new(result_data)
    }
}

impl<T, Ops> Div<T> for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::div_scalar(data, rhs, &mut result_data);
        Self::new(result_data)
    }
}

impl<T, Ops> Div<T> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn div(self, rhs: T) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::div_scalar(data, rhs, &mut result_data);
        NumArray::new(result_data)
    }
}

impl<T, Ops> Div for NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let data = self.get_data();
        let mut result_data = vec![T::default(); data.len()];
        Ops::div_array(data, rhs.get_data(), &mut result_data);
        Self::new(result_data)
    }
}

impl<T, Ops> NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    /// Fallible element-wise division with broadcasting support.
    ///
    /// Returns `Err(BroadcastError)` if shapes are not compatible.
    /// Supports same-shape division and 2D `[m, n] / [m, 1]` broadcasting.
    pub fn try_div_broadcast(&self, rhs: &Self) -> Result<Self, crate::NumError> {
        // Same shape case — SIMD element-wise division
        if self.shape() == rhs.shape() {
            let data = self.get_data();
            let mut result_data = vec![T::default(); data.len()];
            Ops::div_array(data, rhs.get_data(), &mut result_data);
            return Ok(NumArray::new_with_shape(result_data, self.shape().to_vec()));
        }

        // Broadcasting case for 2D arrays: self: [m, n], rhs: [m, 1]
        if self.shape().len() == 2
            && rhs.shape().len() == 2
            && self.shape()[0] == rhs.shape()[0]
            && rhs.shape()[1] == 1
        {
            let (m, n) = (self.shape()[0], self.shape()[1]);
            let mut result_data = Vec::with_capacity(m * n);

            for i in 0..m {
                let divisor = rhs.get(&[i, 0]);
                for j in 0..n {
                    result_data.push(self.get(&[i, j]) / divisor);
                }
            }
            return Ok(NumArray::new_with_shape(result_data, vec![m, n]));
        }

        Err(crate::NumError::BroadcastError {
            lhs: self.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }

    /// Fallible element-wise addition with broadcasting support.
    ///
    /// Supports same-shape addition and 2D `[m, n] + [m, 1]` broadcasting.
    pub fn try_add_broadcast(&self, rhs: &Self) -> Result<Self, crate::NumError> {
        if self.shape() == rhs.shape() {
            let data = self.get_data();
            let mut result_data = vec![T::default(); data.len()];
            Ops::add_array(data, rhs.get_data(), &mut result_data);
            return Ok(NumArray::new_with_shape(result_data, self.shape().to_vec()));
        }

        if self.shape().len() == 2
            && rhs.shape().len() == 2
            && self.shape()[0] == rhs.shape()[0]
            && rhs.shape()[1] == 1
        {
            let (m, n) = (self.shape()[0], self.shape()[1]);
            let mut result_data = Vec::with_capacity(m * n);
            for i in 0..m {
                let addend = rhs.get(&[i, 0]);
                for j in 0..n {
                    result_data.push(self.get(&[i, j]) + addend);
                }
            }
            return Ok(NumArray::new_with_shape(result_data, vec![m, n]));
        }

        Err(crate::NumError::BroadcastError {
            lhs: self.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }

    /// Fallible element-wise subtraction with broadcasting support.
    ///
    /// Supports same-shape subtraction and 2D `[m, n] - [m, 1]` broadcasting.
    pub fn try_sub_broadcast(&self, rhs: &Self) -> Result<Self, crate::NumError> {
        if self.shape() == rhs.shape() {
            let data = self.get_data();
            let mut result_data = vec![T::default(); data.len()];
            Ops::sub_array(data, rhs.get_data(), &mut result_data);
            return Ok(NumArray::new_with_shape(result_data, self.shape().to_vec()));
        }

        if self.shape().len() == 2
            && rhs.shape().len() == 2
            && self.shape()[0] == rhs.shape()[0]
            && rhs.shape()[1] == 1
        {
            let (m, n) = (self.shape()[0], self.shape()[1]);
            let mut result_data = Vec::with_capacity(m * n);
            for i in 0..m {
                let subtrahend = rhs.get(&[i, 0]);
                for j in 0..n {
                    result_data.push(self.get(&[i, j]) - subtrahend);
                }
            }
            return Ok(NumArray::new_with_shape(result_data, vec![m, n]));
        }

        Err(crate::NumError::BroadcastError {
            lhs: self.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }

    /// Fallible element-wise multiplication with broadcasting support.
    ///
    /// Supports same-shape multiplication and 2D `[m, n] * [m, 1]` broadcasting.
    pub fn try_mul_broadcast(&self, rhs: &Self) -> Result<Self, crate::NumError> {
        if self.shape() == rhs.shape() {
            let data = self.get_data();
            let mut result_data = vec![T::default(); data.len()];
            Ops::mul_array(data, rhs.get_data(), &mut result_data);
            return Ok(NumArray::new_with_shape(result_data, self.shape().to_vec()));
        }

        if self.shape().len() == 2
            && rhs.shape().len() == 2
            && self.shape()[0] == rhs.shape()[0]
            && rhs.shape()[1] == 1
        {
            let (m, n) = (self.shape()[0], self.shape()[1]);
            let mut result_data = Vec::with_capacity(m * n);
            for i in 0..m {
                let factor = rhs.get(&[i, 0]);
                for j in 0..n {
                    result_data.push(self.get(&[i, j]) * factor);
                }
            }
            return Ok(NumArray::new_with_shape(result_data, vec![m, n]));
        }

        Err(crate::NumError::BroadcastError {
            lhs: self.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }
}

impl<'b, T, Ops> Div<&'b NumArray<T, Ops>> for &NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    /// # Panics
    /// Panics if shapes are not broadcastable. Use `try_div_broadcast()` for the fallible variant.
    fn div(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        match self.try_div_broadcast(rhs) {
            Ok(result) => result,
            Err(e) => panic!("{}", e),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};

    #[test]
    fn test_add_scalar_u8() {
        let a = NumArrayU8::new(vec![1, 2, 3, 4]);
        let result = a + 1;
        assert_eq!(result.get_data(), &vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_add_array_u8() {
        let a = NumArrayU8::new(vec![1, 2, 3, 4]);
        let b = NumArrayU8::new(vec![4, 3, 2, 1]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5, 5, 5, 5]);
    }

    #[test]
    fn test_add_scalar_i32() {
        let a = NumArrayI32::new(vec![1, 2, 3, 4]);
        let result = a + 1;
        assert_eq!(result.get_data(), &vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_add_scalar_i64() {
        let a = NumArrayI64::new(vec![1, 2, 3, 4]);
        let result = a + 1;
        assert_eq!(result.get_data(), &vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_add_scalar_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = a + 1.0;
        assert_eq!(result.get_data(), &vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_scalar_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = a + 1.0;
        assert_eq!(result.get_data(), &vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_array_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArrayF32::new(vec![4.0, 3.0, 2.0, 1.0]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_add_array_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArrayF64::new(vec![4.0, 3.0, 2.0, 1.0]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_add_scalar_with_remainder() {
        let data = (0..18).map(|x| x as f32).collect::<Vec<_>>(); // Length not divisible by 16 (assuming f32x16)
        let num_array = NumArrayF32::new(data);
        let scalar = 1.0f32;

        let result = num_array + scalar;

        // Check that the result has the correct length
        assert_eq!(result.get_data().len(), 18);

        // Check that each element in the result has been correctly incremented by the scalar
        for (i, &val) in result.get_data().iter().enumerate() {
            assert_eq!(val, i as f32 + scalar);
        }
    }

    #[test]
    fn test_add_arrays_with_remainder() {
        let data_a = (0..18).map(|x| x as f32).collect::<Vec<_>>(); // Length not divisible by 16
        let data_b = (0..18).map(|x| 2.0 * x as f32).collect::<Vec<_>>();

        let num_array_a = NumArrayF32::new(data_a);
        let num_array_b = NumArrayF32::new(data_b);

        let result = num_array_a + num_array_b;

        // Check that the result has the correct length
        assert_eq!(result.get_data().len(), 18);

        // Check that each element in the result is the sum of elements from the original arrays
        for (i, &val) in result.get_data().iter().enumerate() {
            assert_eq!(val, i as f32 + 2.0 * i as f32);
        }
    }

    #[test]
    fn test_broadcast_division() {
        // Test for f32
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = NumArrayF32::new_with_shape(vec![2.0, 4.0], vec![2, 1]);
        let result = &a / &b;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[0.5, 1.0, 1.5, 1.0, 1.25, 1.5]);

        // Test for f64
        let a = NumArrayF64::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = NumArrayF64::new_with_shape(vec![2.0, 4.0], vec![2, 1]);
        let result = &a / &b;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[0.5, 1.0, 1.5, 1.0, 1.25, 1.5]);
    }

    #[test]
    #[should_panic(expected = "shapes not broadcastable")]
    fn test_invalid_broadcast_division() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![2.0], vec![1, 1]);
        let _result = &a / &b; // Should panic
    }

    #[test]
    fn test_broadcast_division_shape_preservation() {
        let a = NumArrayF32::new_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let b = NumArrayF32::new_with_shape(
            vec![14.0_f32.sqrt(), 77.0_f32.sqrt(), 194.0_f32.sqrt()],
            vec![3, 1],
        );
        let result = &a / &b;

        // Check shape is preserved
        assert_eq!(result.shape(), &[3, 3]);

        // Check first row values
        let expected_first_row = [
            1.0 / 14.0_f32.sqrt(),
            2.0 / 14.0_f32.sqrt(),
            3.0 / 14.0_f32.sqrt(),
        ];

        for i in 0..3 {
            assert!(
                (result.get(&[0, i]) - expected_first_row[i]).abs() < 1e-5,
                "Mismatch at position [0, {}]",
                i
            );
        }
    }

    #[test]
    fn test_broadcast_addition() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = NumArrayF32::new_with_shape(vec![10.0, 20.0], vec![2, 1]);
        let result = a.try_add_broadcast(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn test_broadcast_subtraction() {
        let a = NumArrayF32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
        let b = NumArrayF32::new_with_shape(vec![1.0, 2.0], vec![2, 1]);
        let result = a.try_sub_broadcast(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[9.0, 19.0, 29.0, 38.0, 48.0, 58.0]);
    }

    #[test]
    fn test_broadcast_multiplication() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = NumArrayF32::new_with_shape(vec![2.0, 3.0], vec![2, 1]);
        let result = a.try_mul_broadcast(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_broadcast_same_shape() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let result = a.try_add_broadcast(&b).unwrap();
        assert_eq!(result.get_data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_broadcast_incompatible_shapes() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![1.0], vec![1, 1]);
        assert!(a.try_add_broadcast(&b).is_err());
        assert!(a.try_sub_broadcast(&b).is_err());
        assert!(a.try_mul_broadcast(&b).is_err());
    }
}
