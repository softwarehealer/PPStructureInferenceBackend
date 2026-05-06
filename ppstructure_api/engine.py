from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PPStructureConfig:
    root_dir: Path = PROJECT_ROOT
    output_dir: Path | None = None
    det_model_dir: Path | None = None
    rec_model_dir: Path | None = None
    table_model_dir: Path | None = None
    layout_model_dir: Path | None = None
    rec_char_dict_path: Path | None = None
    table_char_dict_path: Path | None = None
    layout_dict_path: Path | None = None
    vis_font_path: Path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    use_gpu: bool = False
    show_log: bool = False

    def resolved(self) -> "PPStructureConfig":
        root = self.root_dir.resolve()
        return PPStructureConfig(
            root_dir=root,
            output_dir=self.output_dir or root / "output",
            det_model_dir=self.det_model_dir or root / "inference/en_PP-OCRv3_det_infer",
            rec_model_dir=self.rec_model_dir or root / "inference/en_PP-OCRv4_rec_infer",
            table_model_dir=self.table_model_dir
            or root / "inference/en_ppstructure_mobile_v2.0_SLANet_infer",
            layout_model_dir=self.layout_model_dir
            or root / "inference/picodet_lcnet_x1_0_fgd_layout_infer",
            rec_char_dict_path=self.rec_char_dict_path or root / "ppocr/utils/en_dict.txt",
            table_char_dict_path=self.table_char_dict_path
            or root / "ppocr/utils/dict/table_structure_dict.txt",
            layout_dict_path=self.layout_dict_path
            or root / "ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
            vis_font_path=self.vis_font_path,
            use_gpu=self.use_gpu,
            show_log=self.show_log,
        )


class PPStructureEngine:
    """Lazy PPStructure wrapper that keeps imports light until inference."""

    def __init__(self, config: PPStructureConfig | None = None) -> None:
        self.config = (config or PPStructureConfig()).resolved()
        self._validate_paths()
        self._structure_system = None

    def predict(self, image: str | os.PathLike[str] | Any) -> list[dict[str, Any]]:
        """Run structure inference and return JSON-safe results.

        `image` may be a filesystem path or an already-loaded OpenCV/NumPy image array.
        Raw cropped images from PaddleOCR are removed from the returned records.
        """
        img = self._load_image(image)
        structure_system = self._get_structure_system()
        result, _time_dict = structure_system(img)
        return self._json_safe(result)

    def predict_file(self, image_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
        return self.predict(image_path)

    def to_json(self, image: str | os.PathLike[str] | Any, **json_kwargs: Any) -> str:
        kwargs = {"ensure_ascii": False, "indent": 2}
        kwargs.update(json_kwargs)
        return json.dumps(self.predict(image), **kwargs)

    def _get_structure_system(self) -> Any:
        if self._structure_system is None:
            from ppstructure.predict_system import StructureSystem

            self._structure_system = StructureSystem(self._build_args())
        return self._structure_system

    def _build_args(self) -> Any:
        from ppstructure.utility import init_args

        parser = init_args()
        args = parser.parse_args([])
        config = self.config

        args.image_dir = None
        args.output = str(config.output_dir)
        args.mode = "structure"
        args.layout = True
        args.table = True
        args.ocr = True
        args.formula = False
        args.recovery = False
        args.use_pdf2docx_api = False
        args.use_gpu = config.use_gpu
        args.show_log = config.show_log
        args.use_angle_cls = False
        args.use_onnx = False
        args.benchmark = False
        args.return_word_box = False

        args.det_model_dir = str(config.det_model_dir)
        args.rec_model_dir = str(config.rec_model_dir)
        args.table_model_dir = str(config.table_model_dir)
        args.layout_model_dir = str(config.layout_model_dir)
        args.rec_char_dict_path = str(config.rec_char_dict_path)
        args.table_char_dict_path = str(config.table_char_dict_path)
        args.layout_dict_path = str(config.layout_dict_path)
        args.vis_font_path = str(config.vis_font_path)
        return args

    def _load_image(self, image: str | os.PathLike[str] | Any) -> Any:
        if not isinstance(image, (str, os.PathLike)):
            return image

        import cv2

        image_path = Path(image).expanduser().resolve()
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img

    def _validate_paths(self) -> None:
        required = {
            "det_model_dir": self.config.det_model_dir,
            "rec_model_dir": self.config.rec_model_dir,
            "table_model_dir": self.config.table_model_dir,
            "layout_model_dir": self.config.layout_model_dir,
            "rec_char_dict_path": self.config.rec_char_dict_path,
            "table_char_dict_path": self.config.table_char_dict_path,
            "layout_dict_path": self.config.layout_dict_path,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Missing PPStructure asset(s):\n" + "\n".join(missing))

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(val)
                for key, val in value.items()
                if key != "img"
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if hasattr(value, "tolist"):
            return cls._json_safe(value.tolist())
        if hasattr(value, "item"):
            return value.item()
        return value
