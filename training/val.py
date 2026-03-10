import argparse
import logging
import tempfile
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.ERROR)


def val(model_path: str, data_path: str, stats_only: bool = False):
    model = YOLO(model_path)

    if stats_only:
        with tempfile.TemporaryDirectory() as tmp:
            results = model.val(
                data=data_path,
                verbose=False,
                plots=False,
                save_txt=False,
                save_json=False,
                save_crop=False,
                save_conf=False,
                project=tmp,
                name='val',
            )
    else:
        results = model.val(data=data_path, verbose=False)

    print(results.results_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a YOLO pose model')
    parser.add_argument('--model', required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--data', required=True, help='Path to dataset config (.yaml file)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Print metrics dict only, skip saving plots/artifacts')
    args = parser.parse_args()

    val(args.model, args.data, stats_only=args.stats_only)
