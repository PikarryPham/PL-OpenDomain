import logging
import os
import numpy as np
import json
import time
import pickle
import datetime
from typing import List, Dict, Any, Union, Tuple

# Thư viện cho TF-IDF và BM25
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Thư viện cho BERT và các biến thể
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
)

# Thư viện cho BMX
logger = logging.getLogger(__name__)

try:
    # Sử dụng baguetter với đường dẫn đúng
    from baguetter.indices.sparse.bmx import BMX, BMXParameters
    from baguetter.indices.sparse.models.bmx.index import SimilarityConf

    BMX_AVAILABLE = True
    logger.info("Baguetter BMX library successfully imported")
except ImportError as e:
    BMX_AVAILABLE = False
    logger.warning(
        f"BMX library not installed. BMX embedding will not be available. Error: {e}"
    )

# Đảm bảo các thư viện được tải
try:
    nltk.download("punkt", quiet=True, download_dir=os.path.expanduser("~/nltk_data"))
    nltk.download(
        "stopwords", quiet=True, download_dir=os.path.expanduser("~/nltk_data")
    )
    # Kiểm tra xem tài nguyên đã được tải thành công chưa
    from nltk.data import find

    try:
        find("tokenizers/punkt")
        find("corpora/stopwords")
        logger.info("NLTK resources successfully loaded")
    except LookupError as e:
        logger.warning(f"NLTK resources not properly loaded: {e}")
        # Thử tải lại với chế độ không im lặng
        logger.info("Retrying download with verbose mode...")
        nltk.download("punkt", quiet=False)
        nltk.download("stopwords", quiet=False)
except Exception as e:
    logger.warning(
        f"Failed to download NLTK resources: {e}. Some functionality may be limited."
    )

# Kích thước của vector OpenAI để đảm bảo tương thích
OPENAI_VECTOR_SIZE = 1536

# Thư mục để lưu các mô hình đã fine-tune
MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
)
os.makedirs(MODELS_DIR, exist_ok=True)

# Khởi tạo các biến toàn cục cho các mô hình và embedders
tfidf_embedder = None
bm25_embedder = None
bmx_embedder = None
roberta_model = None
xlm_roberta_model = None
distilbert_model = None
hybrid_tfidf_bert_model = None
hybrid_bm25_bert_model = None
hybrid_bmx_bert_model = None


# Khởi tạo các mô hình transformer
MODEL_CONFIGS = {
    "bert-base-uncased": {
        "name": "bert-base-uncased",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModel,
    },
    "roberta-base": {
        "name": "roberta-base",
        "tokenizer_class": RobertaTokenizer,
        "model_class": RobertaModel,
    },
    "xlm-roberta-base": {
        "name": "xlm-roberta-base",
        "tokenizer_class": XLMRobertaTokenizer,
        "model_class": XLMRobertaModel,
    },
    "distilbert-base-uncased": {
        "name": "distilbert-base-uncased",
        "tokenizer_class": DistilBertTokenizer,
        "model_class": DistilBertModel,
    },
}

# Cache cho mô hình và tokenizer
loaded_models = {}
loaded_tokenizers = {}


# Khai báo lớp ModelMetadata để lưu thông tin về mô hình
class ModelMetadata:
    def __init__(
        self,
        model_type: str,
        version: str = "1.0",
        created_at: str = None,
        data_size: int = 0,
        performance_metrics: Dict = None,
    ):
        self.model_type = model_type
        self.version = version
        self.created_at = created_at or datetime.datetime.now().isoformat()
        self.data_size = data_size
        self.performance_metrics = performance_metrics or {}

    def to_dict(self) -> Dict:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at,
            "data_size": self.data_size,
            "performance_metrics": self.performance_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetadata":
        return cls(
            model_type=data.get("model_type", "unknown"),
            version=data.get("version", "1.0"),
            created_at=data.get("created_at"),
            data_size=data.get("data_size", 0),
            performance_metrics=data.get("performance_metrics", {}),
        )


# Các tiện ích xử lý văn bản
def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản để chuẩn hóa"""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text


def tokenize_text(text: str) -> List[str]:
    """Tách từ cho văn bản"""
    text = preprocess_text(text)
    try:
        # Thử sử dụng word_tokenize
        tokens = word_tokenize(text)
        # Loại bỏ stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [
            word.lower()
            for word in tokens
            if word.isalnum() and word.lower() not in stop_words
        ]
        return tokens
    except LookupError as e:
        # Nếu xảy ra lỗi liên quan đến punkt_tab, hãy thử cách khác
        logger.warning(f"LookupError in tokenizing text: {e}")
        try:
            # Sử dụng PunktTokenizer trực tiếp nếu có thể
            from nltk.tokenize import PunktTokenizer

            tokenizer = PunktTokenizer()
            tokens = tokenizer.tokenize(text)
            # Loại bỏ stopwords
            try:
                stop_words = set(stopwords.words("english"))
            except:
                stop_words = set()
            tokens = [
                word.lower()
                for word in tokens
                if word.isalnum() and word.lower() not in stop_words
            ]
            return tokens
        except Exception as e2:
            logger.error(f"Second error in tokenizing with PunktTokenizer: {e2}")
            # Nếu vẫn lỗi, quay lại phương pháp đơn giản
            return text.lower().split()
    except Exception as e:
        logger.error(f"General error in tokenizing text: {e}")
        # Trả về một phương pháp tách từ đơn giản nếu có lỗi
        return text.lower().split()


# 1. Mô hình TF-IDF
class TfidfEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=OPENAI_VECTOR_SIZE, tokenizer=tokenize_text, ngram_range=(1, 2)
        )
        self.is_fitted = False
        self.corpus = []
        self.metadata = ModelMetadata(model_type="tfidf")

    def fit(self, corpus: List[str]):
        """Huấn luyện vectorizer trên tập corpus"""
        self.corpus = [preprocess_text(doc) for doc in corpus]
        self.vectorizer.fit(self.corpus)
        self.is_fitted = True
        self.metadata.data_size = len(corpus)
        self.metadata.created_at = datetime.datetime.now().isoformat()
        return self

    def transform(self, text: str) -> np.ndarray:
        """Chuyển đổi văn bản thành vector"""
        if not self.is_fitted:
            # Nếu chưa được huấn luyện, thêm văn bản vào corpus và fit lại
            self.corpus.append(preprocess_text(text))
            self.vectorizer.fit(self.corpus)
            self.is_fitted = True

        text = preprocess_text(text)
        vector = self.vectorizer.transform([text]).toarray()[0]

        # Đảm bảo kích thước vector là OPENAI_VECTOR_SIZE
        if len(vector) < OPENAI_VECTOR_SIZE:
            padding = np.zeros(OPENAI_VECTOR_SIZE - len(vector))
            vector = np.concatenate([vector, padding])
        elif len(vector) > OPENAI_VECTOR_SIZE:
            vector = vector[:OPENAI_VECTOR_SIZE]

        # Chuẩn hóa vector để có độ dài bằng 1
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.astype(np.float32)

    def save(self, version="1.0", performance_metrics=None):
        """Lưu mô hình vào file"""
        if not self.is_fitted:
            logger.warning("Cannot save TF-IDF model: model is not fitted")
            return None

        # Cập nhật metadata
        self.metadata.version = version
        if performance_metrics:
            self.metadata.performance_metrics = performance_metrics

        # Tạo thư mục cho mô hình
        model_dir = os.path.join(MODELS_DIR, "tfidf", version)
        os.makedirs(model_dir, exist_ok=True)

        # Lưu vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        # Lưu metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info(f"TF-IDF model saved to {model_dir}")
        return model_dir

    @classmethod
    def load(cls, version="latest"):
        """Tải mô hình từ file"""
        # Tìm phiên bản mới nhất nếu được yêu cầu
        if version == "latest":
            model_base_dir = os.path.join(MODELS_DIR, "tfidf")
            if not os.path.exists(model_base_dir):
                logger.warning(f"No TF-IDF models found in {model_base_dir}")
                return None

            versions = [
                d
                for d in os.listdir(model_base_dir)
                if os.path.isdir(os.path.join(model_base_dir, d))
            ]
            if not versions:
                logger.warning("No versions found for TF-IDF model")
                return None

            # Sắp xếp theo thời gian tạo
            version = sorted(versions)[-1]

        # Đường dẫn đến thư mục mô hình
        model_dir = os.path.join(MODELS_DIR, "tfidf", version)
        if not os.path.exists(model_dir):
            logger.warning(f"TF-IDF model version {version} not found")
            return None

        # Tải vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            logger.warning(f"Vectorizer file not found at {vectorizer_path}")
            return None

        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        # Tải metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)

        # Tạo và trả về mô hình
        model = cls()
        model.vectorizer = vectorizer
        model.is_fitted = True
        if metadata:
            model.metadata = metadata

        logger.info(f"TF-IDF model loaded from {model_dir}")
        return model


# TF-IDF Embedder instance
tfidf_embedder = TfidfEmbedder()


# Hàm get_embedding cho TF-IDF
def get_embedding_tfidf(text: str) -> List[float]:
    """Hàm tạo embedding sử dụng TF-IDF"""
    text = preprocess_text(text)
    vector = tfidf_embedder.transform(text)
    return vector.tolist()


# 2. Mô hình BM25
class BM25Embedder:
    def __init__(self):
        self.bm25 = None
        self.corpus_tokenized = []
        self.is_fitted = False
        self.metadata = ModelMetadata(model_type="bm25")

    def fit(self, corpus: List[str]):
        """Huấn luyện BM25 trên tập corpus"""
        self.corpus_tokenized = [tokenize_text(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.corpus_tokenized)
        self.is_fitted = True
        self.metadata.data_size = len(corpus)
        self.metadata.created_at = datetime.datetime.now().isoformat()
        return self

    def transform(self, text: str) -> np.ndarray:
        """Chuyển đổi văn bản thành vector dựa trên BM25"""
        if not self.is_fitted or not self.corpus_tokenized:
            # Nếu chưa có corpus, tạo một corpus đơn giản
            self.corpus_tokenized = [tokenize_text(text)]
            self.bm25 = BM25Okapi(self.corpus_tokenized)
            self.is_fitted = True
            # Trong trường hợp này, vector sẽ là vector đơn vị
            vector = np.zeros(OPENAI_VECTOR_SIZE)
            vector[0] = 1.0
            return vector

        text_tokenized = tokenize_text(text)

        # Tính điểm BM25 cho mỗi document trong corpus
        scores = self.bm25.get_scores(text_tokenized)

        # Chuẩn hóa các điểm số
        max_score = max(scores) if scores.any() else 1.0
        normalized_scores = scores / max_score if max_score > 0 else scores

        # Đảm bảo kích thước vector đúng với yêu cầu
        if len(normalized_scores) < OPENAI_VECTOR_SIZE:
            padding = np.zeros(OPENAI_VECTOR_SIZE - len(normalized_scores))
            vector = np.concatenate([normalized_scores, padding])
        else:
            vector = normalized_scores[:OPENAI_VECTOR_SIZE]

        # Chuẩn hóa vector để có độ dài bằng 1
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.astype(np.float32)

    def save(self, version="1.0", performance_metrics=None):
        """Lưu mô hình vào file"""
        if not self.is_fitted:
            logger.warning("Cannot save BM25 model: model is not fitted")
            return None

        # Cập nhật metadata
        self.metadata.version = version
        if performance_metrics:
            self.metadata.performance_metrics = performance_metrics

        # Tạo thư mục cho mô hình
        model_dir = os.path.join(MODELS_DIR, "bm25", version)
        os.makedirs(model_dir, exist_ok=True)

        # Lưu corpus tokenized
        corpus_path = os.path.join(model_dir, "corpus.pkl")
        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus_tokenized, f)

        # Lưu metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info(f"BM25 model saved to {model_dir}")
        return model_dir

    @classmethod
    def load(cls, version="latest"):
        """Tải mô hình từ file"""
        # Tìm phiên bản mới nhất nếu được yêu cầu
        if version == "latest":
            model_base_dir = os.path.join(MODELS_DIR, "bm25")
            if not os.path.exists(model_base_dir):
                logger.warning(f"No BM25 models found in {model_base_dir}")
                return None

            versions = [
                d
                for d in os.listdir(model_base_dir)
                if os.path.isdir(os.path.join(model_base_dir, d))
            ]
            if not versions:
                logger.warning("No versions found for BM25 model")
                return None

            # Sắp xếp theo thời gian tạo
            version = sorted(versions)[-1]

        # Đường dẫn đến thư mục mô hình
        model_dir = os.path.join(MODELS_DIR, "bm25", version)
        if not os.path.exists(model_dir):
            logger.warning(f"BM25 model version {version} not found")
            return None

        # Tải corpus tokenized
        corpus_path = os.path.join(model_dir, "corpus.pkl")
        if not os.path.exists(corpus_path):
            logger.warning(f"Corpus file not found at {corpus_path}")
            return None

        with open(corpus_path, "rb") as f:
            corpus_tokenized = pickle.load(f)

        # Tải metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)

        # Tạo và trả về mô hình
        model = cls()
        model.corpus_tokenized = corpus_tokenized
        model.bm25 = BM25Okapi(corpus_tokenized)
        model.is_fitted = True
        if metadata:
            model.metadata = metadata

        logger.info(f"BM25 model loaded from {model_dir}")
        return model


# BM25 Embedder instance
bm25_embedder = BM25Embedder()


# Hàm get_embedding cho BM25
def get_embedding_bm25(text: str) -> List[float]:
    """Hàm tạo embedding sử dụng BM25"""
    text = preprocess_text(text)
    vector = bm25_embedder.transform(text)
    return vector.tolist()


# 3. Mô hình BMX (nếu thư viện được cài đặt)
class BMXEmbedder:
    def __init__(self):
        self.bmx = None
        self.is_fitted = False
        self.corpus = []
        self.metadata = ModelMetadata(model_type="bmx")

    def fit(self, corpus: List[str]):
        """Huấn luyện BMX trên tập corpus"""
        try:
            if not BMX_AVAILABLE:
                logger.error("Cannot fit BMX model: BMX library is not available")
                return self

            self.corpus = [preprocess_text(doc) for doc in corpus]

            # Cấu hình BMX
            conf = SimilarityConf()
            params = BMXParameters()
            params.set_symmetric(True)
            params.set_default_score(0.0)

            self.bmx = BMX(conf, params)

            # Thêm các document vào BMX
            for i, doc in enumerate(self.corpus):
                self.bmx.add_document(str(i), doc)

            self.is_fitted = True
            self.metadata.data_size = len(corpus)
            self.metadata.created_at = datetime.datetime.now().isoformat()
            logger.info(f"BMX model successfully fitted with {len(corpus)} documents")
            return self
        except Exception as e:
            logger.error(f"Error fitting BMX model: {e}")
            self.is_fitted = False
            return self

    def transform(self, text: str) -> np.ndarray:
        """Chuyển đổi văn bản thành vector dựa trên BMX"""
        try:
            if not BMX_AVAILABLE:
                logger.error("Cannot transform with BMX: BMX library is not available")
                # Trả về vector đơn vị
                vector = np.zeros(OPENAI_VECTOR_SIZE)
                vector[0] = 1.0
                return vector

            if not self.is_fitted:
                # Nếu chưa fit, thử khởi tạo với văn bản đầu vào
                self.corpus = [preprocess_text(text)]

                conf = SimilarityConf()
                params = BMXParameters()
                params.set_symmetric(True)
                params.set_default_score(0.0)

                self.bmx = BMX(conf, params)
                self.bmx.add_document("0", self.corpus[0])
                self.is_fitted = True

                # Trả về vector đơn vị
                vector = np.zeros(OPENAI_VECTOR_SIZE)
                vector[0] = 1.0
                return vector

            text = preprocess_text(text)

            # Tính điểm tương đồng với mỗi document trong corpus
            scores = []
            for i in range(len(self.corpus)):
                score = self.bmx.similarity(text, str(i))
                scores.append(score)

            # Chuyển thành mảng numpy
            scores_array = np.array(scores)

            # Đảm bảo kích thước vector
            if len(scores_array) < OPENAI_VECTOR_SIZE:
                padding = np.zeros(OPENAI_VECTOR_SIZE - len(scores_array))
                vector = np.concatenate([scores_array, padding])
            else:
                vector = scores_array[:OPENAI_VECTOR_SIZE]

            # Chuẩn hóa vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            return vector.astype(np.float32)
        except Exception as e:
            logger.error(f"Error transforming with BMX: {e}")
            # Trả về vector đơn vị nếu có lỗi
            vector = np.zeros(OPENAI_VECTOR_SIZE)
            vector[0] = 1.0
            return vector

    def save(self, version="1.0", performance_metrics=None):
        """Lưu mô hình vào file"""
        if not self.is_fitted:
            logger.warning("Cannot save BMX model: model is not fitted")
            return None

        # Cập nhật metadata
        self.metadata.version = version
        if performance_metrics:
            self.metadata.performance_metrics = performance_metrics

        # Tạo thư mục cho mô hình
        model_dir = os.path.join(MODELS_DIR, "bmx", version)
        os.makedirs(model_dir, exist_ok=True)

        # Lưu corpus
        corpus_path = os.path.join(model_dir, "corpus.pkl")
        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)

        # Lưu metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info(f"BMX model saved to {model_dir}")
        return model_dir

    @classmethod
    def load(cls, version="latest"):
        """Tải mô hình từ file"""
        # Tìm phiên bản mới nhất nếu được yêu cầu
        if version == "latest":
            model_base_dir = os.path.join(MODELS_DIR, "bmx")
            if not os.path.exists(model_base_dir):
                logger.warning(f"No BMX models found in {model_base_dir}")
                return None

            versions = [
                d
                for d in os.listdir(model_base_dir)
                if os.path.isdir(os.path.join(model_base_dir, d))
            ]
            if not versions:
                logger.warning("No versions found for BMX model")
                return None

            # Sắp xếp theo thời gian tạo
            version = sorted(versions)[-1]

        # Đường dẫn đến thư mục mô hình
        model_dir = os.path.join(MODELS_DIR, "bmx", version)
        if not os.path.exists(model_dir):
            logger.warning(f"BMX model version {version} not found")
            return None

        # Tải corpus
        corpus_path = os.path.join(model_dir, "corpus.pkl")
        if not os.path.exists(corpus_path):
            logger.warning(f"Corpus file not found at {corpus_path}")
            return None

        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        # Tải metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)

        # Tạo và trả về mô hình
        model = cls()
        model.fit(corpus)  # Khởi tạo lại BMX với corpus đã lưu
        if metadata:
            model.metadata = metadata

        logger.info(f"BMX model loaded from {model_dir}")
        return model


# BMX Embedder instance nếu thư viện được cài đặt
bmx_embedder = None
if BMX_AVAILABLE:
    try:
        bmx_embedder = BMXEmbedder()
        logger.info("BMX embedder initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize BMX embedder: {str(e)}")
        # Đảm bảo bmx_embedder luôn có giá trị None nếu không khởi tạo được
        bmx_embedder = None
else:
    logger.warning("BMX embedder not available because BMX library is not installed")


# Hàm get_embedding cho BMX
def get_embedding_bmx(text: str) -> List[float]:
    """Hàm tạo embedding sử dụng BMX"""
    if not BMX_AVAILABLE:
        logger.error("BMX embedding not available: BMX library is not installed")
        # Trả về vector zeros nếu BMX không khả dụng
        return [0.0] * OPENAI_VECTOR_SIZE

    if bmx_embedder is None:
        logger.error("BMX embedder is not available")
        # Trả về vector zeros nếu BMX không khả dụng
        return [0.0] * OPENAI_VECTOR_SIZE

    text = preprocess_text(text)
    vector = bmx_embedder.transform(text)
    return vector.tolist()


# Hàm để tải các mô hình cơ bản đã lưu
def load_all_base_models():
    """Tải tất cả các mô hình cơ bản đã lưu"""
    global tfidf_embedder, bm25_embedder, bmx_embedder, BMX_AVAILABLE

    try:
        # Tải mô hình TF-IDF
        tfidf_model = TfidfEmbedder.load()
        if tfidf_model:
            tfidf_embedder = tfidf_model
            logger.info("Loaded TF-IDF model")

        # Tải mô hình BM25
        bm25_model = BM25Embedder.load()
        if bm25_model:
            bm25_embedder = bm25_model
            logger.info("Loaded BM25 model")

        # Tải mô hình BMX
        if BMX_AVAILABLE and bmx_embedder is not None:
            bmx_model = BMXEmbedder.load()
            if bmx_model:
                bmx_embedder = bmx_model
                logger.info("Loaded BMX model")
        else:
            logger.warning("BMX model loading skipped because BMX is not available")

        return True
    except Exception as e:
        logger.error(f"Error loading base models: {e}")
        return False


# 4. Các mô hình dựa trên BERT
# Old version
# def load_transformer_model(
#     model_type, model_name, version="latest", force_download=False
# ):
#     """
#     Tải một phiên bản mô hình transformer cụ thể
#     :param model_type: loại mô hình transformer (roberta, xlm-roberta, distilbert)
#     :param model_name: tên mô hình để tải từ Hugging Face nếu không tìm thấy phiên bản cụ thể
#     :param version: phiên bản muốn tải (hoặc "latest" cho phiên bản mới nhất)
#     :param force_download: Buộc tải lại từ Hugging Face
#     :return: tuple (success, model_info) - success là boolean, model_info là dict chứa thông tin hoặc chuỗi lỗi
#     """
#     global \
#         roberta_model, \
#         xlm_roberta_model, \
#         distilbert_model, \
#         loaded_models, \
#         loaded_tokenizers

#     try:
#         # Kiểm tra loại mô hình hợp lệ
#         if model_type not in ["roberta", "xlm-roberta", "distilbert"]:
#             return (False, f"Unsupported model type: {model_type}")

#         # Thư mục lưu mô hình
#         model_dir = os.path.join(MODELS_DIR, model_type)
#         os.makedirs(model_dir, exist_ok=True)

#         # Khởi tạo biến trước khi sử dụng
#         model_file_exists = False
#         model_exists = False

#         # Xác định phiên bản cụ thể
#         target_version = version
#         if version == "latest":
#             # Liệt kê tất cả thư mục trong models/model_type/ (ngoại trừ thư mục 'latest')
#             all_versions = []
#             try:
#                 all_versions = [
#                     d
#                     for d in os.listdir(model_dir)
#                     if os.path.isdir(os.path.join(model_dir, d)) and d != "latest"
#                 ]
#                 all_versions.sort(
#                     reverse=True
#                 )  # Sắp xếp giảm dần để lấy phiên bản mới nhất
#             except Exception as e:
#                 logger.warning(f"Error listing model versions in {model_dir}: {e}")

#             if all_versions:
#                 target_version = all_versions[0]
#                 logger.info(f"Found latest version for {model_type}: {target_version}")
#             else:
#                 logger.info(
#                     f"No versions found for {model_type}, will download new model"
#                 )
#                 target_version = None

#         # Đường dẫn đến thư mục version
#         version_dir = (
#             os.path.join(model_dir, target_version) if target_version else None
#         )

#         # Kiểm tra sự tồn tại của mô hình
#         model_exists = False
#         if version_dir and os.path.isdir(version_dir):
#             # Kiểm tra các file quan trọng
#             model_file_bin = os.path.join(version_dir, "pytorch_model.bin")
#             model_file_safetensors1 = os.path.join(
#                 version_dir, "pytorch_model.safetensors"
#             )
#             model_file_safetensors2 = os.path.join(version_dir, "model.safetensors")
#             tokenizer_file = os.path.join(version_dir, "tokenizer.json")

#             model_file_exists = (
#                 os.path.exists(model_file_bin)
#                 or os.path.exists(model_file_safetensors1)
#                 or os.path.exists(model_file_safetensors2)
#             )
#             tokenizer_exists = os.path.exists(tokenizer_file)

#             # Mô hình tồn tại nếu có file model
#             model_exists = model_file_exists

#             logger.info(
#                 f"Model check: dir={version_dir}, model_file={model_file_exists} (bin={os.path.exists(model_file_bin)}, safetensors1={os.path.exists(model_file_safetensors1)}, safetensors2={os.path.exists(model_file_safetensors2)}), tokenizer={tokenizer_exists}"
#             )

#         # Quyết định có tải mô hình không
#         should_download = force_download or not model_file_exists

#         model = None
#         tokenizer = None
#         metadata = {}

#         # TRƯỜNG HỢP 1: Tải từ Hugging Face khi force_download hoặc không tìm thấy mô hình cục bộ
#         if should_download:
#             action = (
#                 "Force downloading" if force_download else "Model not found locally"
#             )
#             logger.info(f"{action}, downloading {model_type} from HuggingFace")

#             try:
#                 # Tải mô hình transformer
#                 tokenizer = AutoTokenizer.from_pretrained(model_name)
#                 model = AutoModel.from_pretrained(model_name)

#                 # Tạo thư mục cho phiên bản mới với timestamp
#                 new_version = datetime.datetime.now().strftime("hf_%Y%m%d_%H%M%S")
#                 new_dir = os.path.join(model_dir, new_version)
#                 os.makedirs(new_dir, exist_ok=True)

#                 # Lưu mô hình và tokenizer
#                 model.save_pretrained(new_dir)
#                 tokenizer.save_pretrained(new_dir)

#                 # Tạo metadata
#                 metadata = {
#                     "model_name": model_name,
#                     "model_type": model_type,
#                     "version": new_version,
#                     "created_at": datetime.datetime.now().isoformat(),
#                     "is_quantized": False,
#                     "source": "HuggingFace (downloaded)",
#                     "dimensions": getattr(model.config, "hidden_size", None),
#                 }

#                 # Lưu metadata
#                 with open(os.path.join(new_dir, "metadata.json"), "w") as f:
#                     json.dump(metadata, f, indent=2)

#                 # Cập nhật latest symlink
#                 latest_link = os.path.join(model_dir, "latest")
#                 if os.path.exists(latest_link):
#                     if os.path.islink(latest_link):
#                         os.unlink(latest_link)
#                     elif os.path.isdir(latest_link):
#                         import shutil

#                         shutil.rmtree(latest_link)

#                 # Tạo symlink mới
#                 os.symlink(new_version, latest_link, target_is_directory=True)

#                 # Cập nhật target_version
#                 target_version = new_version
#                 logger.info(f"Successfully downloaded and saved model to {new_dir}")

#             except Exception as e:
#                 logger.error(
#                     f"Caught Exception in load_transformer_model. Type: {type(e)}, Error: {e}",
#                     exc_info=True,
#                 )
#                 return (False, f"Lỗi khi tải mô hình từ Hugging Face: {str(e)}")

#         # TRƯỜNG HỢP 2: Tải từ mô hình cục bộ
#         else:
#             logger.info(f"Loading model from local directory: {version_dir}")

#             try:
#                 # Tải mô hình và tokenizer từ thư mục cục bộ
#                 tokenizer = AutoTokenizer.from_pretrained(version_dir)
#                 model = AutoModel.from_pretrained(version_dir)

#                 # Tải metadata nếu có
#                 metadata_file = os.path.join(version_dir, "metadata.json")
#                 if os.path.exists(metadata_file):
#                     with open(metadata_file, "r") as f:
#                         metadata = json.load(f)
#                 else:
#                     # Tạo metadata cơ bản nếu không có
#                     metadata = {
#                         "model_name": model_name,
#                         "model_type": model_type,
#                         "version": target_version,
#                         "created_at": datetime.datetime.now().isoformat(),
#                         "is_quantized": False,
#                         "source": f"Local ({target_version})",
#                         "dimensions": getattr(model.config, "hidden_size", None),
#                     }

#                 # Phục hồi thuộc tính is_quantized từ metadata
#                 if "is_quantized" in metadata and metadata["is_quantized"]:
#                     model.is_quantized = True
#                     logger.info(
#                         f"Restored quantization state for {model_type} model (is_quantized: {model.is_quantized})"
#                     )

#                     # Kiểm tra xem model đã thực sự được quantize hay chưa
#                     has_quantized_layers = False
#                     for module in model.modules():
#                         if "DynamicQuantizedLinear" in str(type(module)):
#                             has_quantized_layers = True
#                             break

#                     # Nếu model chưa thực sự được quantize nhưng metadata nói rằng đã quantize
#                     if not has_quantized_layers and model.is_quantized:
#                         logger.info(
#                             f"Model {model_type} is marked as quantized in metadata but doesn't have quantized layers. Applying quantization now..."
#                         )
#                         model = quantize_model(model)

#                 logger.info(
#                     f"Successfully loaded model from local directory: {version_dir}"
#                 )

#             except Exception as e:
#                 logger.error(f"Error loading model from {version_dir}: {e}")
#                 return (False, f"Lỗi khi tải mô hình từ thư mục cục bộ: {str(e)}")

#         # Cập nhật biến toàn cục tương ứng
#         if model_type == "roberta":
#             roberta_model = model
#             logger.info(f"Updated global roberta_model: {model is not None}")
#         elif model_type == "xlm-roberta":
#             xlm_roberta_model = model
#             logger.info(f"Updated global xlm_roberta_model: {model is not None}")
#         elif model_type == "distilbert":
#             distilbert_model = model
#             logger.info(f"Updated global distilbert_model: {model is not None}")

#         # Lưu vào cache
#         if model is not None:
#             loaded_models[model_name] = model
#             loaded_tokenizers[model_name] = tokenizer

#         # Kiểm tra một lần nữa xem model đã được gán chưa
#         if model_type == "roberta" and roberta_model is None and model is not None:
#             logger.warning(
#                 "roberta_model is still None after assignment, forcing update"
#             )
#             roberta_model = model
#         elif (
#             model_type == "xlm-roberta"
#             and xlm_roberta_model is None
#             and model is not None
#         ):
#             logger.warning(
#                 "xlm_roberta_model is still None after assignment, forcing update"
#             )
#             xlm_roberta_model = model
#         elif (
#             model_type == "distilbert"
#             and distilbert_model is None
#             and model is not None
#         ):
#             logger.warning(
#                 "distilbert_model is still None after assignment, forcing update"
#             )
#             distilbert_model = model

#         return (True, metadata)
#     except Exception as e:
#         logger.exception(f"Unexpected error in load_transformer_model: {e}")
#         return (False, f"Lỗi không xác định: {str(e)}")


# New version
def load_transformer_model(
    model_type, model_name, version="latest", force_download=False
):
    global \
        roberta_model, \
        xlm_roberta_model, \
        distilbert_model, \
        loaded_models, \
        loaded_tokenizers

    try:
        # Kiểm tra loại mô hình hợp lệ
        if model_type not in ["roberta", "xlm-roberta", "distilbert"]:
            return (False, f"Unsupported model type: {model_type}")

        # Thư mục lưu mô hình
        model_dir = os.path.join(MODELS_DIR, model_type)
        os.makedirs(model_dir, exist_ok=True)

        # Khởi tạo biến trước khi sử dụng
        model_file_exists = False
        model_exists = False

        # Xác định phiên bản cụ thể
        target_version = version
        if version == "latest":
            # Liệt kê tất cả thư mục trong models/model_type/ (ngoại trừ thư mục 'latest')
            all_versions = []
            try:
                all_versions = [
                    d
                    for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d)) and d != "latest"
                ]
                all_versions.sort(
                    reverse=True
                )  # Sắp xếp giảm dần để lấy phiên bản mới nhất
            except Exception as e:
                logger.warning(f"Error listing model versions in {model_dir}: {e}")

            if all_versions:
                target_version = all_versions[0]
                logger.info(f"Found latest version for {model_type}: {target_version}")
            else:
                logger.info(
                    f"No versions found for {model_type}, will download new model"
                )
                target_version = None

        # Đường dẫn đến thư mục version
        version_dir = (
            os.path.join(model_dir, target_version) if target_version else None
        )

        # Đọc metadata từ file trước để biết trạng thái quantization
        metadata = {}
        is_quantized_in_metadata = False
        if version_dir and os.path.isdir(version_dir):
            metadata_path = os.path.join(version_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    is_quantized_in_metadata = metadata.get("is_quantized", False)
                    logger.info(
                        f"Read metadata for {model_type}, is_quantized from metadata: {is_quantized_in_metadata}"
                    )

        # Kiểm tra sự tồn tại của mô hình
        model_exists = False
        if version_dir and os.path.isdir(version_dir):
            # Kiểm tra các file quan trọng
            model_file_bin = os.path.join(version_dir, "pytorch_model.bin")
            model_file_safetensors1 = os.path.join(
                version_dir, "pytorch_model.safetensors"
            )
            model_file_safetensors2 = os.path.join(version_dir, "model.safetensors")
            tokenizer_file = os.path.join(version_dir, "tokenizer.json")

            model_file_exists = (
                os.path.exists(model_file_bin)
                or os.path.exists(model_file_safetensors1)
                or os.path.exists(model_file_safetensors2)
            )
            tokenizer_exists = os.path.exists(tokenizer_file)

            # Mô hình tồn tại nếu có file model
            model_exists = model_file_exists

            logger.info(
                f"Model check: dir={version_dir}, model_file={model_file_exists} (bin={os.path.exists(model_file_bin)}, safetensors1={os.path.exists(model_file_safetensors1)}, safetensors2={os.path.exists(model_file_safetensors2)}), tokenizer={tokenizer_exists}"
            )

        # Quyết định có tải mô hình không
        should_download = force_download or not model_file_exists

        model = None
        tokenizer = None

        # TRƯỜNG HỢP 1: Tải từ Hugging Face khi force_download hoặc không tìm thấy mô hình cục bộ
        if should_download:
            action = (
                "Force downloading" if force_download else "Model not found locally"
            )
            logger.info(f"{action}, downloading {model_type} from HuggingFace")

            try:
                # Tải mô hình transformer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)

                # Tạo thư mục cho phiên bản mới với timestamp
                new_version = datetime.datetime.now().strftime("hf_%Y%m%d_%H%M%S")
                new_dir = os.path.join(model_dir, new_version)
                os.makedirs(new_dir, exist_ok=True)

                # Lưu mô hình và tokenizer
                model.save_pretrained(new_dir)
                tokenizer.save_pretrained(new_dir)

                # Tạo metadata
                metadata = {
                    "model_name": model_name,
                    "model_type": model_type,
                    "version": new_version,
                    "created_at": datetime.datetime.now().isoformat(),
                    "is_quantized": False,
                    "source": "HuggingFace (downloaded)",
                    "dimensions": getattr(model.config, "hidden_size", None),
                }

                # Lưu metadata
                with open(os.path.join(new_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                # Cập nhật latest symlink
                latest_link = os.path.join(model_dir, "latest")
                if os.path.exists(latest_link):
                    if os.path.islink(latest_link):
                        os.unlink(latest_link)
                    elif os.path.isdir(latest_link):
                        import shutil

                        shutil.rmtree(latest_link)

                # Tạo symlink mới
                os.symlink(new_version, latest_link, target_is_directory=True)

                # Cập nhật target_version
                target_version = new_version
                logger.info(f"Successfully downloaded and saved model to {new_dir}")

            except Exception as e:
                logger.error(
                    f"Caught Exception in load_transformer_model. Type: {type(e)}, Error: {e}",
                    exc_info=True,
                )
                return (False, f"Lỗi khi tải mô hình từ Hugging Face: {str(e)}")

        # TRƯỜNG HỢP 2: Tải từ mô hình cục bộ
        else:
            logger.info(f"Loading model from local directory: {version_dir}")

            try:
                # Tải mô hình và tokenizer từ thư mục cục bộ
                tokenizer = AutoTokenizer.from_pretrained(version_dir)
                model = AutoModel.from_pretrained(version_dir)

                # Nếu model được đánh dấu là đã quantized trong metadata, thiết lập thuộc tính này
                if is_quantized_in_metadata:
                    model.is_quantized = True
                    logger.info(
                        f"Set is_quantized=True for {model_type} model from metadata"
                    )

                    # Kiểm tra xem model đã thực sự được quantize hay chưa
                    has_quantized_layers = False
                    for module in model.modules():
                        if "DynamicQuantizedLinear" in str(type(module)):
                            has_quantized_layers = True
                            break

                    # Nếu model chưa thực sự được quantize nhưng metadata nói rằng đã quantize
                    if not has_quantized_layers and model.is_quantized:
                        logger.info(
                            f"Model {model_type} is marked as quantized in metadata but doesn't have quantized layers. Applying quantization now..."
                        )
                        model = quantize_model(model)

                logger.info(
                    f"Successfully loaded model from local directory: {version_dir}"
                )

            except Exception as e:
                logger.error(f"Error loading model from {version_dir}: {e}")
                return (False, f"Lỗi khi tải mô hình từ thư mục cục bộ: {str(e)}")

        # Cập nhật biến toàn cục tương ứng
        if model_type == "roberta":
            roberta_model = model
            logger.info(f"Updated global roberta_model: {model is not None}")
        elif model_type == "xlm-roberta":
            xlm_roberta_model = model
            logger.info(f"Updated global xlm_roberta_model: {model is not None}")
        elif model_type == "distilbert":
            distilbert_model = model
            logger.info(f"Updated global distilbert_model: {model is not None}")

        # Lưu vào cache
        if model is not None:
            loaded_models[model_name] = model
            loaded_tokenizers[model_name] = tokenizer

        return (True, metadata)
    except Exception as e:
        logger.exception(f"Unexpected error in load_transformer_model: {e}")
        return (False, f"Lỗi không xác định: {str(e)}")


def mean_pooling(model_output, attention_mask):
    """Tính trung bình của embeddings từ layer cuối của BERT"""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_bert_embedding(text: str, model_name: str) -> List[float]:
    """Tạo embedding sử dụng mô hình BERT"""
    try:
        # Xác định loại mô hình
        model_type = model_name.split("-")[0]
        if model_type == "xlm":
            model_type = "xlm-roberta"

        # Tải mô hình và tokenizer
        if model_name in loaded_models and model_name in loaded_tokenizers:
            # Sử dụng từ cache nếu đã tải
            tokenizer = loaded_tokenizers[model_name]
            model = loaded_models[model_name]
        else:
            # Tải mô hình từ Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            # Cache lại để sử dụng sau
            loaded_tokenizers[model_name] = tokenizer
            loaded_models[model_name] = model

            # Cập nhật biến global
            if model_type == "roberta":
                roberta_model = model
            elif model_type == "xlm-roberta":
                xlm_roberta_model = model
            elif model_type == "distilbert":
                distilbert_model = model

        # Mã hóa văn bản
        encoded_input = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Tắt gradient để tăng tốc độ
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Tính pooling trên output của model
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Chuẩn hóa vector
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )

        # Chuyển đổi tensor thành numpy array
        embedding = sentence_embeddings[0].numpy()

        # Điều chỉnh kích thước vector
        if embedding.shape[0] < OPENAI_VECTOR_SIZE:
            padding = np.zeros(OPENAI_VECTOR_SIZE - embedding.shape[0])
            embedding = np.concatenate([embedding, padding])
        elif embedding.shape[0] > OPENAI_VECTOR_SIZE:
            embedding = embedding[:OPENAI_VECTOR_SIZE]

        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error getting BERT embedding: {e}")
        # Trả về vector zeros nếu có lỗi
        return [0.0] * OPENAI_VECTOR_SIZE


# Các hàm get_embedding cho từng loại mô hình BERT
def get_embedding_roberta(text: str) -> List[float]:
    """Hàm tạo embedding sử dụng RoBERTa-base"""
    return get_bert_embedding(text, "roberta-base")


def get_embedding_xlm_roberta(text: str) -> List[float]:
    """Hàm tạo embedding sử dụng XLM-RoBERTa-base"""
    return get_bert_embedding(text, "xlm-roberta-base")


def get_embedding_distilbert(text: str) -> List[float]:
    """Hàm tạo embedding sử dụng DistilBERT"""
    return get_bert_embedding(text, "distilbert-base-uncased")


# 5. Kết hợp các mô hình
def get_embedding_hybrid_tfidf_bert(text: str) -> List[float]:
    """Kết hợp TF-IDF và BERT (DistilBERT)"""
    tfidf_vector = np.array(get_embedding_tfidf(text))
    bert_vector = np.array(get_embedding_distilbert(text))

    # Trọng số cho kết hợp (có thể điều chỉnh)
    tfidf_weight = 0.3
    bert_weight = 0.7

    # Kết hợp hai vector
    combined_vector = tfidf_weight * tfidf_vector + bert_weight * bert_vector

    # Chuẩn hóa vector kết hợp
    norm = np.linalg.norm(combined_vector)
    if norm > 0:
        combined_vector = combined_vector / norm

    return combined_vector.tolist()


def get_embedding_hybrid_bm25_bert(text: str) -> List[float]:
    """Kết hợp BM25 và BERT (DistilBERT)"""
    bm25_vector = np.array(get_embedding_bm25(text))
    bert_vector = np.array(get_embedding_distilbert(text))

    # Trọng số cho kết hợp (có thể điều chỉnh)
    bm25_weight = 0.3
    bert_weight = 0.7

    # Kết hợp hai vector
    combined_vector = bm25_weight * bm25_vector + bert_weight * bert_vector

    # Chuẩn hóa vector kết hợp
    norm = np.linalg.norm(combined_vector)
    if norm > 0:
        combined_vector = combined_vector / norm

    return combined_vector.tolist()


def get_embedding_hybrid_bmx_bert(text: str) -> List[float]:
    """Kết hợp BMX và BERT (DistilBERT)"""
    bmx_vector = np.array(get_embedding_bmx(text))
    bert_vector = np.array(get_embedding_distilbert(text))

    # Trọng số cho kết hợp (có thể điều chỉnh)
    bmx_weight = 0.3
    bert_weight = 0.7

    # Kết hợp hai vector
    combined_vector = bmx_weight * bmx_vector + bert_weight * bert_vector

    # Chuẩn hóa vector kết hợp
    norm = np.linalg.norm(combined_vector)
    if norm > 0:
        combined_vector = combined_vector / norm

    return combined_vector.tolist()


# Từ điển chứa tất cả các mô hình embedding
embedding_models = {
    "openai": "get_embedding",  # Hàm gốc từ OpenAI (không thay đổi)
    "tfidf": get_embedding_tfidf,
    "bm25": get_embedding_bm25,
    "bmx": get_embedding_bmx,
    "roberta": get_embedding_roberta,
    "xlm-roberta": get_embedding_xlm_roberta,
    "distilbert": get_embedding_distilbert,
    "hybrid_tfidf_bert": get_embedding_hybrid_tfidf_bert,
    "hybrid_bm25_bert": get_embedding_hybrid_bm25_bert,
    "hybrid_bmx_bert": get_embedding_hybrid_bmx_bert,
}


# Chức năng đánh giá và so sánh các mô hình
def compare_embedding_models(
    query_texts: List[str], corpus_texts: List[str], model_names: List[str] = None
) -> Dict[str, Any]:
    """
    So sánh các mô hình embedding khác nhau

    Args:
        query_texts: Danh sách các văn bản truy vấn
        corpus_texts: Danh sách các văn bản trong corpus
        model_names: Danh sách tên các mô hình cần so sánh

    Returns:
        Kết quả so sánh các mô hình
    """
    if model_names is None:
        model_names = list(embedding_models.keys())

    # Kết quả so sánh
    results = {"models": {}, "summary": {}}

    # Huấn luyện các mô hình cần corpus
    tfidf_embedder.fit(corpus_texts)
    bm25_embedder.fit(corpus_texts)
    if bmx_embedder is not None:
        try:
            bmx_embedder.fit(corpus_texts)
        except:
            logger.warning("Could not fit BMX embedder")

    # So sánh từng mô hình
    for model_name in model_names:
        if model_name not in embedding_models:
            logger.warning(f"Model {model_name} not found in available models")
            continue

        logger.info(f"Evaluating model: {model_name}")
        model_results = {
            "query_time": [],
            "corpus_time": [],
        }

        # Thời gian tạo embedding cho corpus
        start_time = time.time()
        for text in corpus_texts:
            if model_name == "openai":
                # Đây là trường hợp đặc biệt khi ta không gọi trực tiếp hàm OpenAI
                pass
            else:
                embedding_models[model_name](text)
        corpus_time = time.time() - start_time
        model_results["corpus_time"] = (
            corpus_time / len(corpus_texts) if corpus_texts else 0
        )

        # Thời gian tạo embedding cho các truy vấn
        start_time = time.time()
        for text in query_texts:
            if model_name == "openai":
                # Đây là trường hợp đặc biệt khi ta không gọi trực tiếp hàm OpenAI
                pass
            else:
                embedding_models[model_name](text)
        query_time = time.time() - start_time
        model_results["query_time"] = (
            query_time / len(query_texts) if query_texts else 0
        )

        # Lưu kết quả
        results["models"][model_name] = model_results

    return results


# Hàm lưu kết quả so sánh
def save_comparison_results(results: Dict[str, Any], output_file: str):
    """Lưu kết quả so sánh các mô hình ra file"""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Comparison results saved to {output_file}")


# Hàm fine-tuning mô hình - cập nhật để lưu mô hình sau khi fine-tune
def fine_tune_tfidf(corpus_texts: List[str], version="1.0", save_model=True):
    """Fine-tune mô hình TF-IDF trên corpus cụ thể và lưu lại"""
    tfidf_embedder.fit(corpus_texts)

    # Lưu mô hình nếu được yêu cầu
    if save_model:
        performance_metrics = {"corpus_size": len(corpus_texts)}
        tfidf_embedder.save(version=version, performance_metrics=performance_metrics)

    return tfidf_embedder


def fine_tune_bm25(corpus_texts: List[str], version="1.0", save_model=True):
    """Fine-tune mô hình BM25 trên corpus cụ thể và lưu lại"""
    bm25_embedder.fit(corpus_texts)

    # Lưu mô hình nếu được yêu cầu
    if save_model:
        performance_metrics = {"corpus_size": len(corpus_texts)}
        bm25_embedder.save(version=version, performance_metrics=performance_metrics)

    return bm25_embedder


def fine_tune_bmx(corpus_texts: List[str], version="1.0", save_model=True):
    """Fine-tune mô hình BMX trên corpus cụ thể và lưu lại"""
    if bmx_embedder is not None:
        bmx_embedder.fit(corpus_texts)

        # Lưu mô hình nếu được yêu cầu
        if save_model:
            performance_metrics = {"corpus_size": len(corpus_texts)}
            bmx_embedder.save(version=version, performance_metrics=performance_metrics)

    return bmx_embedder


def fine_tune_transformer_model(
    model_name, corpus_texts, version="1.0", save_model=True
):
    """
    Fine-tune một mô hình transformer trên dữ liệu văn bản
    :param model_name: Tên mô hình transformer (roberta-base, xlm-roberta-base, distilbert-base-uncased)
    :param corpus_texts: Danh sách các văn bản huấn luyện
    :param version: Phiên bản của mô hình sau khi fine-tune
    :param save_model: Có lưu mô hình không
    :return: Mô hình đã được fine-tune
    """
    import torch
    from datetime import datetime
    from transformers import AutoTokenizer, AutoModel
    from pathlib import Path
    import os
    import json
    import shutil

    try:
        # Tạo thư mục cho mô hình
        model_type = model_name.split("-")[0]
        if model_type == "xlm":
            model_type = "xlm-roberta"

        model_dir = os.path.join(MODELS_DIR, model_type)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Xác định phiên bản cụ thể
        target_version = version
        if version == "latest":
            # Liệt kê tất cả thư mục trong models/model_type/ (ngoại trừ thư mục 'latest')
            all_versions = []
            try:
                all_versions = [
                    d
                    for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d)) and d != "latest"
                ]
                all_versions.sort(
                    reverse=True
                )  # Sắp xếp giảm dần để lấy phiên bản mới nhất
            except Exception as e:
                logger.warning(f"Error listing model versions in {model_dir}: {e}")

            if all_versions:
                target_version = all_versions[0]
                logger.info(f"Found latest version for {model_type}: {target_version}")
            else:
                logger.info(
                    f"No versions found for {model_type}, will download new model"
                )
                target_version = None

        # Đường dẫn đến thư mục version
        version_dir = (
            os.path.join(model_dir, target_version) if target_version else None
        )

        # Kiểm tra sự tồn tại của mô hình
        model_exists = False
        if version_dir and os.path.isdir(version_dir):
            # Kiểm tra các file quan trọng
            model_file_bin = os.path.join(version_dir, "pytorch_model.bin")
            model_file_safetensors1 = os.path.join(
                version_dir, "pytorch_model.safetensors"
            )
            model_file_safetensors2 = os.path.join(version_dir, "model.safetensors")
            tokenizer_file = os.path.join(version_dir, "tokenizer.json")

            model_file_exists = (
                os.path.exists(model_file_bin)
                or os.path.exists(model_file_safetensors1)
                or os.path.exists(model_file_safetensors2)
            )
            tokenizer_exists = os.path.exists(tokenizer_file)

            # Mô hình tồn tại nếu cả hai loại file đều tồn tại
            model_exists = model_file_exists

            logger.info(
                f"Model check: dir={version_dir}, model_file={model_file_exists} (bin={os.path.exists(model_file_bin)}, safetensors1={os.path.exists(model_file_safetensors1)}, safetensors2={os.path.exists(model_file_safetensors2)}), tokenizer={tokenizer_exists}"
            )

        # Quyết định có tải mô hình không
        should_download = not model_exists

        model = None
        tokenizer = None
        metadata = {}

        # TRƯỜNG HỢP 1: Tải từ Hugging Face khi force_download hoặc không tìm thấy mô hình cục bộ
        if should_download:
            action = "Model not found locallyyyyy"
            logger.info(f"{action}, downloading {model_type} from HuggingFace")

            try:
                # Tải mô hình transformer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)

                # Tạo thư mục cho phiên bản mới với timestamp
                new_version = datetime.datetime.now().strftime("hf_%Y%m%d_%H%M%S")
                new_dir = os.path.join(model_dir, new_version)
                os.makedirs(new_dir, exist_ok=True)

                # Lưu mô hình và tokenizer
                model.save_pretrained(new_dir)
                tokenizer.save_pretrained(new_dir)

                # Tạo metadata
                metadata = {
                    "model_name": model_name,
                    "model_type": model_type,
                    "version": new_version,
                    "created_at": datetime.datetime.now().isoformat(),
                    "is_quantized": False,
                    "source": "HuggingFace (downloaded)",
                    "dimensions": getattr(model.config, "hidden_size", None),
                }

                # Lưu metadata
                with open(os.path.join(new_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                # Cập nhật latest symlink
                latest_link = os.path.join(model_dir, "latest")
                if os.path.exists(latest_link):
                    if os.path.islink(latest_link):
                        os.unlink(latest_link)
                    elif os.path.isdir(latest_link):
                        import shutil

                        shutil.rmtree(latest_link)

                # Tạo symlink mới
                os.symlink(new_version, latest_link, target_is_directory=True)

                # Cập nhật target_version
                target_version = new_version
                logger.info(f"Successfully downloaded and saved model to {new_dir}")

            except Exception as e:
                logger.error(
                    f"Caught Exception in load_transformer_model. Type: {type(e)}, Error: {e}",
                    exc_info=True,
                )
                return (False, f"Lỗi khi tải mô hình từ Hugging Face: {str(e)}")

        # TRƯỜNG HỢP 2: Tải từ mô hình cục bộ
        else:
            logger.info(f"Loading model from local directory: {version_dir}")

            try:
                # Tải mô hình và tokenizer từ thư mục cục bộ
                tokenizer = AutoTokenizer.from_pretrained(version_dir)
                model = AutoModel.from_pretrained(version_dir)

                # Tải metadata nếu có
                metadata_file = os.path.join(version_dir, "metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                else:
                    # Tạo metadata cơ bản nếu không có
                    metadata = {
                        "model_name": model_name,
                        "model_type": model_type,
                        "version": target_version,
                        "created_at": datetime.datetime.now().isoformat(),
                        "is_quantized": False,
                        "source": f"Local ({target_version})",
                        "dimensions": getattr(model.config, "hidden_size", None),
                    }

                # Phục hồi thuộc tính is_quantized từ metadata
                if "is_quantized" in metadata and metadata["is_quantized"]:
                    model.is_quantized = True
                    logger.info(
                        f"Restored quantization state for {model_type} model (is_quantized: {model.is_quantized})"
                    )

                    # Kiểm tra xem model đã thực sự được quantize hay chưa
                    has_quantized_layers = False
                    for module in model.modules():
                        if "DynamicQuantizedLinear" in str(type(module)):
                            has_quantized_layers = True
                            break

                    # Nếu model chưa thực sự được quantize nhưng metadata nói rằng đã quantize
                    if not has_quantized_layers and model.is_quantized:
                        logger.info(
                            f"Model {model_type} is marked as quantized in metadata but doesn't have quantized layers. Applying quantization now..."
                        )
                        model = quantize_model(model)

                logger.info(
                    f"Successfully loaded model from local directory: {version_dir}"
                )

            except Exception as e:
                logger.error(f"Error loading model from {version_dir}: {e}")
                return (False, f"Lỗi khi tải mô hình từ thư mục cục bộ: {str(e)}")

        # Cập nhật biến toàn cục tương ứng
        if model_type == "roberta":
            roberta_model = model
        elif model_type == "xlm-roberta":
            xlm_roberta_model = model
        elif model_type == "distilbert":
            distilbert_model = model

        # Lưu vào cache
        loaded_models[model_name] = model
        loaded_tokenizers[model_name] = tokenizer

        # Hoàn thành và trả về thông tin
        model_info = {
            "model_name": model_name,
            "model_type": model_type,
            "version": target_version,
            "created_at": metadata.get(
                "created_at", datetime.datetime.now().isoformat()
            ),
            "is_quantized": getattr(model, "is_quantized", False),
            "source": metadata.get("source", "unknown"),
            "dimensions": getattr(model.config, "hidden_size", None),
        }

        return (True, model_info)

    except Exception as e:
        logger.exception(f"Unexpected error in load_transformer_model: {e}")
        return (False, f"Lỗi không xác định: {str(e)}")


def quantize_model(model):
    """
    Áp dụng quantization cho mô hình transformer để giảm kích thước
    :param model: Mô hình transformer
    :return: Mô hình đã được quantize hoặc None nếu có lỗi
    """
    global roberta_model, xlm_roberta_model, distilbert_model

    try:
        # Kiểm tra model có tồn tại không
        if model is None:
            logger.error("Model is None, cannot quantize")
            return None

        # Kiểm tra xem model có phải là PyTorch model hay không
        if not hasattr(model, "eval") or not callable(getattr(model, "eval")):
            logger.error("Invalid model object, missing eval() method")
            return None

        # Kiểm tra nếu model đã được quantize
        if hasattr(model, "is_quantized") and model.is_quantized:
            logger.info("Model is already quantized, skipping...")
            return model

        # Chỉ quantize nếu PyTorch hỗ trợ
        if hasattr(torch, "quantization"):
            logger.info("Applying quantization to model...")

            # Chuyển mô hình sang chế độ đánh giá
            model.eval()

            # Áp dụng dynamic quantization cho các lớp Linear
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

            # Đánh dấu mô hình đã được quantize
            quantized_model.is_quantized = True

            # Xác định loại model để cập nhật metadata
            model_type = None
            if model is roberta_model:
                model_type = "roberta"
            elif model is xlm_roberta_model:
                model_type = "xlm-roberta"
            elif model is distilbert_model:
                model_type = "distilbert"

            if model_type:
                logger.info(f"Saving quantization metadata for {model_type} model")
                model_dir = os.path.join(MODELS_DIR, model_type)
                if os.path.exists(model_dir):
                    versions = [
                        d
                        for d in os.listdir(model_dir)
                        if os.path.isdir(os.path.join(model_dir, d)) and d != "latest"
                    ]
                    if versions:
                        versions.sort(reverse=True)
                        latest_version = versions[0]
                        metadata_path = os.path.join(
                            model_dir, latest_version, "metadata.json"
                        )
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, "r") as f:
                                    metadata = json.load(f)
                                metadata["is_quantized"] = True
                                with open(metadata_path, "w") as f:
                                    json.dump(metadata, f, indent=2)
                                logger.info(
                                    f"Updated metadata for {model_type} with quantization info"
                                )

                                # Cập nhật file latest/metadata.json nếu tồn tại
                                latest_metadata_path = os.path.join(
                                    model_dir, "latest", "metadata.json"
                                )
                                if os.path.exists(latest_metadata_path):
                                    try:
                                        with open(latest_metadata_path, "r") as f:
                                            latest_metadata = json.load(f)
                                        latest_metadata["is_quantized"] = True
                                        with open(latest_metadata_path, "w") as f:
                                            json.dump(latest_metadata, f, indent=2)
                                        logger.info(
                                            f"Updated latest metadata for {model_type} with quantization info"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Error updating latest metadata for {model_type}: {e}"
                                        )
                            except Exception as e:
                                logger.error(
                                    f"Error updating metadata for {model_type}: {e}"
                                )

            # Cập nhật biến toàn cục
            if model_type == "roberta":
                roberta_model = quantized_model
            elif model_type == "xlm-roberta":
                xlm_roberta_model = quantized_model
            elif model_type == "distilbert":
                distilbert_model = quantized_model

            logger.info("Quantization completed successfully")
            return quantized_model
        else:
            logger.info("PyTorch quantization not available, skipping...")
            return model
    except Exception as e:
        logger.error(f"Error quantizing model: {e}", exc_info=True)
        return model


def fine_tune_hybrid_model(
    trad_model_type,
    transformer_model_name,
    corpus_texts,
    version="1.0",
    save_model=True,
):
    """
    Fine-tune một mô hình hybrid (kết hợp mô hình truyền thống và transformer)
    :param trad_model_type: Loại mô hình truyền thống (tfidf, bm25, bmx)
    :param transformer_model_name: Tên mô hình transformer
    :param corpus_texts: Danh sách các văn bản huấn luyện
    :param version: Phiên bản của mô hình sau khi fine-tune
    :param save_model: Có lưu mô hình không
    :return: Mô hình đã được fine-tune
    """
    from datetime import datetime
    import os
    import json
    from pathlib import Path

    try:
        # Fine-tune mô hình truyền thống
        trad_model = None
        if trad_model_type == "tfidf":
            trad_model = fine_tune_tfidf(
                corpus_texts, version=version, save_model=save_model
            )
        elif trad_model_type == "bm25":
            trad_model = fine_tune_bm25(
                corpus_texts, version=version, save_model=save_model
            )
        elif trad_model_type == "bmx":
            trad_model = fine_tune_bmx(
                corpus_texts, version=version, save_model=save_model
            )
        else:
            raise ValueError(f"Unknown traditional model type: {trad_model_type}")

        # Fine-tune mô hình transformer
        transformer_type = transformer_model_name.split("-")[0]
        if transformer_type == "xlm":
            transformer_type = "xlm-roberta"

        transformer_model = fine_tune_transformer_model(
            transformer_model_name, corpus_texts, version=version, save_model=save_model
        )

        # Lưu thông tin kết hợp
        hybrid_model_type = f"hybrid_{trad_model_type}_{transformer_type}"
        hybrid_model_dir = os.path.join(MODELS_DIR, hybrid_model_type)
        Path(hybrid_model_dir).mkdir(parents=True, exist_ok=True)

        version_dir = os.path.join(hybrid_model_dir, version)
        Path(version_dir).mkdir(exist_ok=True)

        # Lưu metadata
        if save_model:
            metadata = {
                "base_models": [trad_model_type, transformer_model_name],
                "version": version,
                "created_at": datetime.now().isoformat(),
                "data_size": len(corpus_texts),
                "trad_model_info": trad_model.metadata.to_dict()
                if hasattr(trad_model, "metadata")
                else {},
                "transformer_model_info": {
                    "model_name": transformer_model_name,
                    "is_quantized": hasattr(transformer_model, "is_quantized")
                    and transformer_model.is_quantized,
                },
            }

            with open(os.path.join(version_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            # Lưu thông tin liên kết đến các mô hình thành phần
            with open(os.path.join(version_dir, "component_models.json"), "w") as f:
                json.dump(
                    {
                        "trad_model": {"type": trad_model_type, "version": version},
                        "transformer_model": {
                            "type": transformer_type,
                            "name": transformer_model_name,
                            "version": version,
                        },
                    },
                    f,
                    indent=2,
                )

        # Cập nhật biến toàn cục
        transformer_model = None
        # Xác định loại transformer model từ tên
        transformer_type = transformer_model_name.split("-")[0]
        if transformer_type == "xlm":
            transformer_type = "xlm-roberta"

        if transformer_type == "roberta":
            transformer_model = roberta_model
        elif transformer_type == "xlm-roberta":
            transformer_model = xlm_roberta_model
        elif transformer_type == "distilbert":
            transformer_model = distilbert_model
        elif transformer_type == "bert":
            from transformers import AutoModel

            transformer_model = AutoModel.from_pretrained(transformer_model_name)

        hybrid_model = {
            "trad_model": trad_model,
            "transformer_model": transformer_model,
        }

        if hybrid_model_type == "hybrid_tfidf_bert":
            hybrid_tfidf_bert_model = hybrid_model
        elif hybrid_model_type == "hybrid_bm25_bert":
            hybrid_bm25_bert_model = hybrid_model
        elif hybrid_model_type == "hybrid_bmx_bert":
            hybrid_bmx_bert_model = hybrid_model

        return {"trad_model": trad_model, "transformer_model": transformer_model}
    except Exception as e:
        logger.error(
            f"Error fine-tuning hybrid model {trad_model_type} + {transformer_model_name}: {e}"
        )
        raise


def load_hybrid_model(trad_model_type, transformer_model_name, version="latest"):
    """
    Tải mô hình hybrid đã lưu
    :param trad_model_type: Loại mô hình truyền thống (tfidf, bm25, bmx)
    :param transformer_model_name: Tên mô hình transformer
    :param version: Phiên bản muốn tải
    :return: (success, model_info)
    """
    global hybrid_tfidf_bert_model, hybrid_bm25_bert_model, hybrid_bmx_bert_model

    try:
        hybrid_type = f"hybrid_{trad_model_type}_bert"
        hybrid_dir = os.path.join(MODELS_DIR, hybrid_type)

        # Kiểm tra thư mục tồn tại
        if not os.path.exists(hybrid_dir):
            os.makedirs(hybrid_dir, exist_ok=True)

        # Tìm phiên bản mới nhất nếu cần
        if version == "latest":
            versions = []
            if os.path.exists(hybrid_dir):
                versions = [
                    d
                    for d in os.listdir(hybrid_dir)
                    if os.path.isdir(os.path.join(hybrid_dir, d))
                ]

            if not versions:
                logger.warning(f"No versions found for {hybrid_type} model")
                # Khởi tạo mô hình mới từ các mô hình thành phần
                return create_new_hybrid_model(trad_model_type, transformer_model_name)

            # Tìm phiên bản mới nhất
            try:
                versions.sort(
                    key=lambda v: json.load(
                        open(os.path.join(hybrid_dir, v, "metadata.json"), "r")
                    ).get("created_at", ""),
                    reverse=True,
                )
            except:
                versions.sort(reverse=True)

            version = versions[0]

        # Kiểm tra phiên bản tồn tại
        version_dir = os.path.join(hybrid_dir, version)
        if not os.path.exists(version_dir):
            logger.warning(f"Hybrid model {hybrid_type} version {version} not found")
            return (False, f"Version {version} not found")

        # Tải metadata
        metadata_path = os.path.join(version_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "trad_model": trad_model_type,
                "transformer_model": transformer_model_name,
                "version": version,
            }

        # Tạo đối tượng hybrid model
        hybrid_model = {
            "trad_model_type": trad_model_type,
            "transformer_model": transformer_model_name,
            "metadata": metadata,
        }

        # Cập nhật biến global tương ứng
        if trad_model_type == "tfidf":
            hybrid_tfidf_bert_model = hybrid_model
            logger.info(f"Loaded hybrid TF-IDF+BERT model version {version}")
        elif trad_model_type == "bm25":
            hybrid_bm25_bert_model = hybrid_model
            logger.info(f"Loaded hybrid BM25+BERT model version {version}")
        elif trad_model_type == "bmx":
            hybrid_bmx_bert_model = hybrid_model
            logger.info(f"Loaded hybrid BMX+BERT model version {version}")

        return (True, metadata)
    except Exception as e:
        logger.error(f"Error loading hybrid model: {e}")
        return (False, str(e))


def create_new_hybrid_model(trad_model_type, transformer_model_name):
    """
    Tạo mô hình hybrid mới khi không tìm thấy phiên bản nào
    """
    global hybrid_tfidf_bert_model, hybrid_bm25_bert_model, hybrid_bmx_bert_model

    try:
        hybrid_type = f"hybrid_{trad_model_type}_bert"

        # Tạo metadata
        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "trad_model": trad_model_type,
            "transformer_model": transformer_model_name,
            "version": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        # Tạo đối tượng hybrid model
        hybrid_model = {
            "trad_model_type": trad_model_type,
            "transformer_model": transformer_model_name,
            "metadata": metadata,
        }

        # Lưu metadata
        hybrid_dir = os.path.join(MODELS_DIR, hybrid_type)
        os.makedirs(hybrid_dir, exist_ok=True)

        version_dir = os.path.join(hybrid_dir, metadata["version"])
        os.makedirs(version_dir, exist_ok=True)

        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Cập nhật biến global tương ứng
        if trad_model_type == "tfidf":
            hybrid_tfidf_bert_model = hybrid_model
            logger.info(
                f"Created new hybrid TF-IDF+BERT model version {metadata['version']}"
            )
        elif trad_model_type == "bm25":
            hybrid_bm25_bert_model = hybrid_model
            logger.info(
                f"Created new hybrid BM25+BERT model version {metadata['version']}"
            )
        elif trad_model_type == "bmx":
            hybrid_bmx_bert_model = hybrid_model
            logger.info(
                f"Created new hybrid BMX+BERT model version {metadata['version']}"
            )

        return True, metadata
    except Exception as e:
        logger.error(f"Error creating hybrid model: {e}")
        return False, str(e)


# Tự động tải các mô hình khi module được import
try:
    load_all_base_models()
except Exception as e:
    logger.error(f"Error loading base models: {e}")
    logger.info("Continuing with default models")
