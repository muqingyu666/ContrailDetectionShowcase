"""Small script to run the contrail detection demo."""
import argparse
import numpy as np
from contrail_detection import (
    load_and_preprocess,
    multi_angle_convolution_detection,
    visualize_results,
)


def main():
    parser = argparse.ArgumentParser(description="Run contrail detection demo")
    parser.add_argument(
        "image",
        default="contrail_showcase_1.png",
        help="Path to image for detection",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive matplotlib windows",
    )
    args = parser.parse_args()

    original_img, preprocessed = load_and_preprocess(args.image)
    angles = np.arange(0, 180, 15)
    max_response, binary_detection, angle_responses, angle_map, max_angle_idx = (
        multi_angle_convolution_detection(preprocessed, angles=angles, kernel_size=21)
    )

    if args.show:
        visualize_results(
            original_img,
            preprocessed,
            max_response,
            binary_detection,
            angle_responses,
            angle_map,
            angles,
            max_angle_idx,
        )
    else:
        # Save overlay images without showing them
        visualize_results(
            original_img,
            preprocessed,
            max_response,
            binary_detection,
            angle_responses,
            angle_map,
            angles,
            max_angle_idx,
        )


if __name__ == "__main__":
    main()
