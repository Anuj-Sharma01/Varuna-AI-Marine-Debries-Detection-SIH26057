import cv2
import numpy as np


def clahe_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast normalization to a sonar image.

    Supports both grayscale and BGR images.
    Geometry is preserved:
    - No resizing
    - No cropping
    - No rotation
    """

    if image is None:
        raise ValueError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    if len(image.shape) == 2:
        # XTF sonar data is single-channel.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(image)

    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Support normal BGR images too.
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l_channel = clahe.apply(l_channel)

        enhanced = cv2.merge(
            (l_channel, a_channel, b_channel)
        )

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_LAB2BGR
        )

    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

def handle_dropouts(image: np.ndarray) -> np.ndarray:
    """
    Safely handle isolated sonar dropout rows.

    Large continuous zero regions are preserved because they may
    represent normal no-return sonar regions.
    """

    result = image.copy()

    if len(result.shape) != 2:
        return result

    zero_ratio = (result == 0).mean(axis=1)
    dropout_rows = (
        (zero_ratio > 0.95)
        & (np.roll(zero_ratio > 0.95, 1) == False)
        & (np.roll(zero_ratio > 0.95, -1) == False)
    )

    for row in np.where(dropout_rows)[0]:
        if row == 0 or row == result.shape[0] - 1:
            continue

        result[row] = (
            (
                result[row - 1].astype(np.uint16)
                + result[row + 1].astype(np.uint16)
            ) // 2
        ).astype(np.uint8)

    return result
    
def nlm_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast normalization followed by
    Non-Local Means denoising.

    Supports grayscale sonar images.
    Geometry is preserved:
    - No resizing
    - No cropping
    - No rotation
    """

    image = handle_dropouts(image)
    image = clahe_preprocess(image)
    
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            10,
            7,
            21
        )

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )


def preprocess_image(input_path: str, output_path: str) -> None:
    """
    Read an image, apply NLM preprocessing, and save the result.
    """

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    processed = nlm_preprocess(image)

    success = cv2.imwrite(output_path, processed)

    if not success:
        raise IOError(f"Could not write image: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply NLM preprocessing to a sonar image."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input sonar image."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for preprocessed image."
    )

    args = parser.parse_args()

    preprocess_image(args.input, args.output)

    print(f"Preprocessed image saved to: {args.output}")