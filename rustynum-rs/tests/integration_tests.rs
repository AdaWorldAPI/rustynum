use rustynum_rs::{NumArrayF32, NumArrayF64, NumError};

#[test]
fn test_num_array_creation_and_dot_product() {
    // Create two `NumArray` instances with test data.
    let array1 = NumArrayF32::from(&[1.0, 2.0, 3.0, 4.0][..]);
    let array2 = NumArrayF32::from(vec![4.0, 3.0, 2.0, 1.0]);

    // Perform a dot product operation between the two arrays.
    let result = array1.dot(&array2);

    // Check that the dot product result is as expected.
    assert_eq!(
        result.get_data(),
        &[20.0],
        "The dot product of the two arrays should be 20.0"
    );

    // Test with arrays of different sizes to ensure it handles non-multiples of SIMD width
    let array3 = NumArrayF32::from(&[1.0, 2.0, 3.0][..]);
    let array4 = NumArrayF32::from(vec![4.0, 5.0, 6.0]);

    // Perform a dot product operation between the two smaller arrays.
    let result_small = array3.dot(&array4);

    // Check that the dot product result is as expected for the smaller arrays.
    assert_eq!(
        result_small.get_data(),
        &[32.0],
        "The dot product of the two smaller arrays should be 32.0"
    );
}

#[test]
fn test_complex_operations() {
    let size = 1000; // Choose a size for the vectors
    let constant = 2.5f64;

    // Generate two large NumArray instances
    let data1: Vec<f64> = (0..size).map(|x| x as f64).collect();

    let step = size as f64 / (size - 1) as f64; // Correct step calculation
    let data2: Vec<f64> = (0..size).map(|x| size as f64 - x as f64 * step).collect();

    let array1 = NumArrayF64::from(data1.clone());
    let array2 = NumArrayF64::from(data2.clone());

    // Perform addition of two NumArrays
    let added = &array1 + &array2;

    // Subtract the mean from the added array
    let mean = added.mean().item();
    let subtracted_mean = &added - mean;

    // Divide by a constant value
    let divided = &subtracted_mean / constant;

    // Multiply with another NumArray (using the original array1 for simplicity)
    let multiplied = &divided * &array1;

    // Perform a dot product with the initial input (array2)
    let dot_product_result = multiplied.dot(&array2).item();
    // Print arrray2

    // Expected result calculation using ndarray or manual calculation
    // Placeholder for expected result
    let expected_result = 0.000012200523883620917; // Calculate the expected result

    let tolerance = 1e-1; // Define a suitable tolerance for your scenario
    let actual_error = (dot_product_result - expected_result).abs();
    assert!(
        actual_error <= tolerance,
        "The complex operation result does not match the expected value. \
        Expected: {}, Actual: {}, Error: {}",
        expected_result,
        dot_product_result,
        actual_error
    );
}

// ============================================================================
// Edge-case tests: empty arrays
// ============================================================================

#[test]
fn test_empty_array_mean() {
    let a = NumArrayF32::new(vec![]);
    // mean of empty array returns 0/0 = NaN
    let m = a.mean().item();
    assert!(m.is_nan(), "mean of empty array should be NaN, got {}", m);
}

#[test]
fn test_empty_array_dot() {
    let a = NumArrayF32::new(vec![]);
    let b = NumArrayF32::new(vec![]);
    let d = a.dot(&b);
    assert_eq!(d.get_data(), &[0.0], "dot of empty arrays should be 0");
}

// ============================================================================
// Edge-case tests: NaN propagation
// ============================================================================

#[test]
fn test_nan_propagation_sum() {
    let a = NumArrayF32::new(vec![1.0, f32::NAN, 3.0]);
    let s: f32 = a.mean().item();
    assert!(s.is_nan(), "mean with NaN input should propagate NaN");
}

#[test]
fn test_nan_propagation_dot() {
    let a = NumArrayF32::new(vec![1.0, f32::NAN, 3.0]);
    let b = NumArrayF32::new(vec![4.0, 5.0, 6.0]);
    let d = a.dot(&b).item();
    assert!(d.is_nan(), "dot with NaN input should propagate NaN");
}

// ============================================================================
// Edge-case tests: single-element arrays
// ============================================================================

#[test]
fn test_single_element_operations() {
    let a = NumArrayF32::new(vec![42.0]);
    let b = NumArrayF32::new(vec![2.0]);

    assert_eq!(a.mean().item(), 42.0);
    assert_eq!(a.median().item(), 42.0);
    assert_eq!(a.dot(&b).item(), 84.0);
    assert_eq!((&a + &b).get_data(), &[44.0]);
    assert_eq!((&a * &b).get_data(), &[84.0]);
}

// ============================================================================
// Edge-case tests: fallible API (try_*)
// ============================================================================

#[test]
fn test_try_reshape_mismatch() {
    let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = a.try_reshape(&[2, 4]); // 2*4=8 != 6
    assert!(result.is_err());
    let err = result.err().unwrap();
    match err {
        NumError::ShapeMismatch {
            data_len,
            shape_product,
        } => {
            assert_eq!(data_len, 6);
            assert_eq!(shape_product, 8);
        }
        e => panic!("Expected ShapeMismatch, got {:?}", e),
    }
}

#[test]
fn test_try_reshape_ok() {
    let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = a.try_reshape(&[2, 3]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().shape(), &[2, 3]);
}

#[test]
fn test_try_transpose_non_2d() {
    let a = NumArrayF32::new(vec![1.0, 2.0, 3.0]);
    let result = a.try_transpose();
    assert!(result.is_err());
}

#[test]
fn test_try_transpose_ok() {
    let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = a.try_transpose();
    assert!(result.is_ok());
    let t = result.unwrap();
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.get_data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_try_slice_axis_oob() {
    let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = a.try_slice(5, 0, 1); // axis 5 on 2D
    assert!(result.is_err());
    let err = result.err().unwrap();
    match err {
        NumError::AxisOutOfBounds { axis, ndim } => {
            assert_eq!(axis, 5);
            assert_eq!(ndim, 2);
        }
        e => panic!("Expected AxisOutOfBounds, got {:?}", e),
    }
}

#[test]
fn test_try_arange_zero_step() {
    let result = NumArrayF32::try_arange(0.0, 10.0, 0.0);
    assert!(result.is_err());
    let err = result.err().unwrap();
    match err {
        NumError::InvalidParameter(_) => {}
        e => panic!("Expected InvalidParameter, got {:?}", e),
    }
}

#[test]
fn test_try_matrix_multiply_dimension_mismatch() {
    use rustynum_rs::num_array::linalg::try_matrix_multiply;

    let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let result = try_matrix_multiply(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_try_matrix_multiply_ok() {
    use rustynum_rs::num_array::linalg::try_matrix_multiply;

    let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = NumArrayF32::new_with_shape(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let result = try_matrix_multiply(&a, &b);
    assert!(result.is_ok());
    let c = result.unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get_data(), &[58.0, 64.0, 139.0, 154.0]);
}

// ============================================================================
// Edge-case tests: large arrays (SIMD boundary crossing)
// ============================================================================

#[test]
fn test_large_array_dot_product() {
    // 1025 elements: crosses SIMD lane boundaries (16, 32, 64 lanes)
    let n = 1025;
    let a = NumArrayF32::new(vec![1.0; n]);
    let b = NumArrayF32::new(vec![2.0; n]);
    let result = a.dot(&b).item();
    assert_eq!(result, 2050.0, "dot of 1025 ones * 2s should be 2050");
}

#[test]
fn test_f64_precision() {
    // Kahan-sensitive: sum of many small numbers
    let n = 100_000;
    let val = 1e-8_f64;
    let a = NumArrayF64::new(vec![val; n]);
    let expected = val * n as f64;
    let result = a.mean().item() * n as f64;
    let rel_err = (result - expected).abs() / expected;
    assert!(
        rel_err < 1e-6,
        "f64 sum of 100k * 1e-8 should be accurate, rel_err={}",
        rel_err
    );
}
