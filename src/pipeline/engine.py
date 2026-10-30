from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PaddleOCR import ppstructure  as pps
from PaddleOCR.ppstructure.predict_system import StructureSystem
from PaddleOCR.ppstructure.utility import init_args
from configs.settings import settings
from configs import app_logger as logger

@dataclass(frozen=True)
class PPStructureConfig:
    root_dir: Path = Path(settings.PPSTRUCTURE_ROOT_DIR)
    output_dir: Path = Path(settings.PPSTRUCTURE_OUTPUT_DIR)
    det_model_dir: Path = Path(settings.PPSTRUCTURE_DET_MODEL_DIR)
    rec_model_dir: Path = Path(settings.PPSTRUCTURE_REC_MODEL_DIR)
    table_model_dir: Path = Path(settings.PPSTRUCTURE_TABLE_MODEL_DIR)
    layout_model_dir: Path = Path(settings.PPSTRUCTURE_LAYOUT_MODEL_DIR)
    rec_char_dict_path: Path = Path(settings.PPSTRUCTURE_REC_CHAR_DICT_PATH)
    table_char_dict_path: Path = Path(settings.PPSTRUCTURE_TABLE_CHAR_DICT_PATH)
    layout_dict_path: Path = Path(settings.PPSTRUCTURE_LAYOUT_DICT_PATH)
    vis_font_path: Path = Path(settings.PPSTRUCTURE_VIS_FONT_PATH)
    use_gpu: bool = settings.USE_GPU
    show_log: bool = settings.PPSTRUCTURE_SHOW_LOG
    det_limit_side_len: float = settings.PPSTRUCTURE_DET_LIMIT_SIDE_LEN
    det_db_thresh: float = settings.PPSTRUCTURE_DET_DB_THRESH
    det_db_box_thresh: float = settings.PPSTRUCTURE_DET_DB_BOX_THRESH
    det_db_unclip_ratio: float = settings.PPSTRUCTURE_DET_DB_UNCLIP_RATIO
    drop_score: float = settings.PPSTRUCTURE_DROP_SCORE
    rec_batch_num: int = settings.PPSTRUCTURE_REC_BATCH_NUM
    ir_optim: bool = settings.PPSTRUCTURE_IR_OPTIM
    use_tensorrt: bool = settings.PPSTRUCTURE_USE_TENSORRT
    precision: str = settings.PPSTRUCTURE_PRECISION

    def resolved(self) -> "PPStructureConfig":
        root = self.root_dir.resolve()
        logger.info(f">>>>>> root={root}")
        return PPStructureConfig(
            root_dir=root,
            output_dir=self.output_dir or root / "output",
            det_model_dir=self.det_model_dir or f"{settings.INFERENCE_BASE_PATH}/en_PP-OCRv3_det_infer",
            rec_model_dir=self.rec_model_dir or f"{settings.INFERENCE_BASE_PATH}/en_PP-OCRv4_rec_infer",
            table_model_dir=self.table_model_dir
            or  f"{settings.INFERENCE_BASE_PATH}/en_ppstructure_mobile_v2.0_SLANet_infer",
            layout_model_dir=self.layout_model_dir
            or f"{settings.INFERENCE_BASE_PATH}/picodet_lcnet_x1_0_fgd_layout_infer",
            rec_char_dict_path=self.rec_char_dict_path or f"{root}/PaddleOCR/ppocr/utils/en_dict.txt",
            table_char_dict_path=self.table_char_dict_path
            or f"{root}/PaddleOCR/ppocr/utils/dict/table_structure_dict.txt",
            layout_dict_path=self.layout_dict_path
            or f"{root}/PaddleOCR/ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
            vis_font_path=self.vis_font_path,
            use_gpu=settings.USE_GPU,
            show_log=settings.PPSTRUCTURE_SHOW_LOG,
            det_limit_side_len=settings.PPSTRUCTURE_DET_LIMIT_SIDE_LEN,
            det_db_thresh=settings.PPSTRUCTURE_DET_DB_THRESH,
            det_db_box_thresh=settings.PPSTRUCTURE_DET_DB_BOX_THRESH,
            det_db_unclip_ratio=settings.PPSTRUCTURE_DET_DB_UNCLIP_RATIO,
            drop_score=settings.PPSTRUCTURE_DROP_SCORE,
            rec_batch_num=settings.PPSTRUCTURE_REC_BATCH_NUM,
            ir_optim=settings.PPSTRUCTURE_IR_OPTIM,
            use_tensorrt=settings.PPSTRUCTURE_USE_TENSORRT,
            precision=settings.PPSTRUCTURE_PRECISION,
        )


class PPStructureEngine:
    """Lazy PPStructure wrapper that keeps imports light until inference."""

    def __init__(self, config: PPStructureConfig | None = None) -> None:
        self.config = (config or PPStructureConfig()).resolved()
        self._validate_paths()
        self._structure_system = None

    def predict(self, image: str | os.PathLike[str] | Any, *, word_boxes: bool = False) -> list[dict[str, Any]]:
        """Run structure inference and return JSON-safe results.

        `image` may be a filesystem path or an already-loaded OpenCV/NumPy image array.
        Raw cropped images from PaddleOCR are removed from the returned records.
        With `word_boxes`, each OCR line also carries `text_word` and `text_word_region`.
        """
        img = self._load_image(image)
        structure_system = self._get_structure_system()
        self._set_word_boxes(structure_system, word_boxes)
        result, _time_dict = structure_system(img)
        return self._json_safe(result)

    def predict_file(self, image_path: str | os.PathLike[str], *, word_boxes: bool = False) -> list[dict[str, Any]]:
        return self.predict(image_path, word_boxes=word_boxes)

    @staticmethod
    def _set_word_boxes(structure_system: Any, enabled: bool) -> None:
        # PaddleOCR reads return_word_box from these attributes on every call, so it can be switched
        # per request without reloading models. Measured cost when enabled is within noise.
        structure_system.return_word_box = enabled
        if structure_system.text_system is not None:
            structure_system.text_system.text_recognizer.return_word_box = enabled

    def to_json(self, image: str | os.PathLike[str] | Any, **json_kwargs: Any) -> str:
        kwargs = {"ensure_ascii": False, "indent": 2}
        kwargs.update(json_kwargs)
        return json.dumps(self.predict(image), **kwargs)

    def _get_structure_system(self) -> Any:
        if self._structure_system is None:
            self._structure_system = StructureSystem(self._build_args())
        return self._structure_system

    def _build_args(self) -> Any:
        """
        Function to prepare arguments for PPStructure OCR call
        """
        parser = init_args()
        args = parser.parse_args([])
        config = self.config

        args.image_dir = settings.PPSTRUCTURE_IMAGE_DIR or None
        args.output = str(config.output_dir)
        args.mode = settings.PPSTRUCTURE_MODE
        args.layout = settings.PPSTRUCTURE_LAYOUT
        args.table = settings.PPSTRUCTURE_TABLE
        args.ocr = settings.PPSTRUCTURE_OCR
        args.formula = settings.PPSTRUCTURE_FORMULA
        args.recovery = settings.PPSTRUCTURE_RECOVERY
        args.use_pdf2docx_api = settings.PPSTRUCTURE_USE_PDF2DOCX_API
        args.use_gpu = config.use_gpu
        args.show_log = config.show_log
        args.use_angle_cls = settings.PPSTRUCTURE_USE_ANGLE_CLS
        args.cls_model_dir = settings.PPSTRUCTURE_CLS_MODEL_DIR or None
        args.cls_thresh = settings.PPSTRUCTURE_CLS_THRESH
        args.use_onnx = settings.PPSTRUCTURE_USE_ONNX
        args.benchmark = settings.PPSTRUCTURE_BENCHMARK
        args.return_word_box = settings.PPSTRUCTURE_RETURN_WORD_BOX
        args.det_limit_side_len = config.det_limit_side_len
        args.det_db_thresh = config.det_db_thresh
        args.det_db_box_thresh = config.det_db_box_thresh
        args.det_db_unclip_ratio = config.det_db_unclip_ratio
        args.drop_score = config.drop_score
        args.rec_batch_num = config.rec_batch_num
        args.ir_optim = config.ir_optim
        args.use_tensorrt = config.use_tensorrt
        args.precision = config.precision

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
        if settings.PPSTRUCTURE_USE_ANGLE_CLS:
            required["cls_model_dir"] = settings.PPSTRUCTURE_CLS_MODEL_DIR
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
