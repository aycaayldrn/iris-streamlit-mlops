import pytest
import numpy as np
from app import predict

# Example: features for a Setosa (known to be class 0)
TEST_FEATURES = [5.1, 3.5, 1.4, 0.2]


def test_prediction_returns_valid_class_id():
    """
    Tests that the predict function returns a valid integer class ID (0, 1, or 2).
    """
    # 1. Call the prediction function with the test data
    result = predict(TEST_FEATURES)

    # 2. Check that the result is not an error string
    assert not isinstance(result, str), f"Prediction failed with error: {result}"

    # 3. Check the type: The prediction returns a numpy array, we need the first element.
    assert isinstance(result, np.ndarray), "Prediction result is not a NumPy array."

    # 4. Check the value: The returned class index must be one of the three valid species IDs.
    # We check the first (and only) element of the result array.
    class_id = result[0]

    # Valid Iris species IDs are 0, 1, or 2
    assert class_id in [0, 1, 2], f"Invalid class ID returned: {class_id}"

    # Optional: Verify the result is an integer
    assert isinstance(class_id, np.int64), "Predicted class ID is not an integer type."
