from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PPS_PIPELINE_API_PATH_PREFIX: str = "/"
    PPS_PIPELINE_API_ROOT_PATH: str = "/"
    
    LOG_LEVEL: int
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    INFERENCE_BASE_PATH: str
    FONT_PATH: str
    USE_GPU: bool

    PPSTRUCTURE_ROOT_DIR: str = "src"
    PPSTRUCTURE_OUTPUT_DIR: str = "src/output"
    PPSTRUCTURE_DET_MODEL_DIR: str
    PPSTRUCTURE_REC_MODEL_DIR: str
    PPSTRUCTURE_TABLE_MODEL_DIR: str
    PPSTRUCTURE_LAYOUT_MODEL_DIR: str
    PPSTRUCTURE_REC_CHAR_DICT_PATH: str
    PPSTRUCTURE_TABLE_CHAR_DICT_PATH: str
    PPSTRUCTURE_LAYOUT_DICT_PATH: str
    PPSTRUCTURE_VIS_FONT_PATH: str

    PPSTRUCTURE_IMAGE_DIR: str = ""
    PPSTRUCTURE_MODE: str = "structure"
    PPSTRUCTURE_LAYOUT: bool = True
    PPSTRUCTURE_TABLE: bool = True
    PPSTRUCTURE_OCR: bool = True
    PPSTRUCTURE_FORMULA: bool = False
    PPSTRUCTURE_RECOVERY: bool = False
    PPSTRUCTURE_USE_PDF2DOCX_API: bool = False
    PPSTRUCTURE_SHOW_LOG: bool = False
    PPSTRUCTURE_USE_ANGLE_CLS: bool = False
    # Text direction classifier model; needed when PPSTRUCTURE_USE_ANGLE_CLS is true (fixes upside-down and vertical text)
    PPSTRUCTURE_CLS_MODEL_DIR: str = ""
    # Minimum confidence to flip a text line the classifier thinks is upside down; Paddle's 0.9 flips some normal lines
    PPSTRUCTURE_CLS_THRESH: float = 0.95
    PPSTRUCTURE_USE_ONNX: bool = False
    PPSTRUCTURE_BENCHMARK: bool = False
    PPSTRUCTURE_RETURN_WORD_BOX: bool = False
    PPSTRUCTURE_DET_LIMIT_SIDE_LEN: float = 960
    PPSTRUCTURE_DET_DB_THRESH: float = 0.3
    PPSTRUCTURE_DET_DB_BOX_THRESH: float = 0.6
    PPSTRUCTURE_DET_DB_UNCLIP_RATIO: float = 1.5
    PPSTRUCTURE_DROP_SCORE: float = 0.5
    PPSTRUCTURE_REC_BATCH_NUM: int = 6
    PPSTRUCTURE_IR_OPTIM: bool = True
    PPSTRUCTURE_USE_TENSORRT: bool = False
    PPSTRUCTURE_PRECISION: str = "fp32"

    # Minimum fraction of an OCR line's area that must lie inside a requested box to count as a match
    EXTRACTION_MIN_OVERLAP: float = 0.5

    # Gemini drafts a template for a document no stored template matches
    GEMINI_API_KEY: SecretStr = SecretStr("")
    GEMINI_MODEL_NAME: str = "gemini-3-flash-preview"
    # Seconds per attempt; drafting reads the whole page and all its OCR words
    GEMINI_TIMEOUT: int = 180
    GEMINI_MAX_RETRIES: int = 3

    model_config=SettingsConfigDict(env_file=".env")

settings = Settings()
