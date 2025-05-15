import re
import logging
import torch
import numpy as np
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForSequenceClassification,
)
from embeddings import get_embedding_distilbert, load_transformer_model

logger = logging.getLogger(__name__)

# Khởi tạo model summarization và topic extraction
SUMMARIZER = None
TOPIC_EXTRACTOR = None
NLP = None

# Danh sách các chủ đề phổ biến để phân loại
COMMON_TOPICS = [
    "Art",
    "History",
    "Science",
    "Technology",
    "Sports",
    "Entertainment",
    "Politics",
    "Business",
    "Education",
    "Health",
    "Travel",
    "Fashion",
    "Music",
    "Literature",
    "Food",
    "Environment",
    "Religion",
    "Philosophy",
    "Mathematics",
    "Biology",
    "Physics",
    "Chemistry",
    "Psychology",
    "Computer Science",
    "Economics",
    "Geography",
    "Sociology",
    "Anthropology",
    "Linguistics",
    "Architecture",
    "Engineering",
    "Medicine",
    "Law",
]


def init_models():
    """Khởi tạo các mô hình cần thiết cho summarization và topic extraction"""
    global SUMMARIZER, TOPIC_EXTRACTOR, NLP

    try:
        # Load spaCy model
        logger.info("Loading spaCy model for text processing...")
        NLP = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        logger.info("Trying to download spaCy model...")
        try:
            import os

            os.system("python -m spacy download en_core_web_sm")
            NLP = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {e}")
            NLP = None

    try:
        # Load summarization model (DistilBART-CNN)
        logger.info("Loading summarization model (DistilBART-CNN)...")
        summarizer_name = "sshleifer/distilbart-cnn-12-6"
        summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_name)
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name)

        # Quantize model để giảm kích thước và tăng hiệu suất
        if torch.cuda.is_available():
            summarizer_model = summarizer_model.to("cuda")
        else:
            # Quantize để giảm dùng bộ nhớ
            summarizer_model = torch.quantization.quantize_dynamic(
                summarizer_model, {torch.nn.Linear}, dtype=torch.qint8
            )

        SUMMARIZER = pipeline(
            "summarization",
            model=summarizer_model,
            tokenizer=summarizer_tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        logger.info("Summarization model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading summarization model: {e}")
        logger.info("Trying alternative summarization model (t5-small)...")
        try:
            summarizer_name = "t5-small"
            SUMMARIZER = pipeline(
                "summarization",
                model=summarizer_name,
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("Alternative summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load alternative summarization model: {e}")
            SUMMARIZER = None

    try:
        # Load Zero-shot classification model for topic extraction
        logger.info("Loading Zero-shot classification model for topic extraction...")
        topic_model_name = "facebook/bart-large-mnli"

        # Thử tận dụng mô hình DistilBERT có sẵn
        success, _ = load_transformer_model("distilbert", "distilbert-base-uncased")
        if success:
            logger.info("Using existing DistilBERT model for topic extraction")
            TOPIC_EXTRACTOR = "distilbert-base-uncased"
        else:
            # Nếu không có sẵn, dùng Zero-shot classification
            topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_name)
            topic_model = AutoModelForSequenceClassification.from_pretrained(
                topic_model_name
            )

            # Quantize model
            if torch.cuda.is_available():
                topic_model = topic_model.to("cuda")
            else:
                topic_model = torch.quantization.quantize_dynamic(
                    topic_model, {torch.nn.Linear}, dtype=torch.qint8
                )

            TOPIC_EXTRACTOR = pipeline(
                "zero-shot-classification",
                model=topic_model,
                tokenizer=topic_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("Zero-shot classification model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading topic extraction model: {e}")
        TOPIC_EXTRACTOR = None


def summarize_text(text: str, max_length: int = 150, min_length: int = 50) -> str:
    """
    Tóm tắt nội dung văn bản sử dụng mô hình transformer

    Args:
        text: Văn bản cần tóm tắt
        max_length: Độ dài tối đa của bản tóm tắt
        min_length: Độ dài tối thiểu của bản tóm tắt

    Returns:
        Văn bản đã được tóm tắt
    """
    if not text:
        logger.warning("Empty text provided for summarization")
        return ""

    # Đầu tiên, kiểm tra xem đây có phải là đối thoại tiếng Việt không
    if _is_vietnamese_dialogue(text):
        logger.info(
            "Vietnamese dialogue detected, using specialized dialogue summarization"
        )
        return _summarize_vietnamese_dialogue(text)

    # Điều chỉnh min_length để tránh trả về nguyên văn bản quá ngắn
    if len(text.split()) < 10:  # Nếu văn bản rất ngắn
        logger.warning(
            "Text is too short for summarization, returning with minimal processing"
        )
        return text

    # Điều chỉnh min_length dựa trên độ dài văn bản để tránh trường hợp summary = input
    adjusted_min_length = min(min_length, max(10, len(text.split()) // 3))
    adjusted_max_length = min(
        max_length, max(adjusted_min_length + 5, len(text.split()) // 2)
    )

    if adjusted_max_length <= adjusted_min_length:
        adjusted_max_length = adjusted_min_length + 10

    logger.info(
        f"Adjusted length params: min={adjusted_min_length}, max={adjusted_max_length}"
    )

    if SUMMARIZER is None:
        logger.warning("Summarization model not initialized, initializing now...")
        init_models()

    if SUMMARIZER is None:
        logger.error("Failed to initialize summarization model")
        # Trường hợp không tìm thấy mô hình, thử tóm tắt đơn giản
        return _fallback_summarize(text)

    try:
        # Tiền xử lý văn bản đầu vào để loại bỏ dấu hiệu có thể gây nhiễu
        preprocessed_text = text

        # Xử lý trường hợp có quá nhiều số hoặc ký hiệu đặc biệt lặp lại (như trong công thức nấu ăn)
        import re

        # Loại bỏ các số lặp lại nhiều lần
        preprocessed_text = re.sub(r"(\d+)(\s*\1\s*)+", r"\1", preprocessed_text)

        # Loại bỏ các từ lặp lại nhiều lần liên tiếp - sử dụng regex mạnh hơn
        preprocessed_text = re.sub(r"(\b\w+\b)(\s+\1\b)+", r"\1", preprocessed_text)

        # Thêm bước loại bỏ các từ lặp lại trong toàn bộ văn bản (không chỉ liên tiếp)
        for common_word in [
            "egg",
            "eggs",
            "cream",
            "oven",
            "ready",
            "sugar",
            "g",
            "ml",
        ]:
            # Tìm các từ lặp lại quá nhiều lần (>3 lần)
            if preprocessed_text.lower().count(" " + common_word + " ") > 3:
                # Đếm số lần xuất hiện
                count = preprocessed_text.lower().count(" " + common_word + " ")
                logger.info(
                    f"Found '{common_word}' repeated {count} times, reducing repetition"
                )

                # Thay thế các từ lặp lại bằng một mẫu ít lặp lại hơn
                parts = preprocessed_text.split(".")
                for i, part in enumerate(parts):
                    # Chỉ giữ lại 1 lần xuất hiện của từ trong mỗi câu
                    if part.lower().count(" " + common_word + " ") > 1:
                        # Tạo một mẫu regex để tìm và thay thế từ lặp lại
                        pattern = (
                            r"(\s+"
                            + common_word
                            + r"\s+).*?(\s+"
                            + common_word
                            + r"\s+)"
                        )
                        parts[i] = re.sub(pattern, r"\1", part, flags=re.IGNORECASE)

                preprocessed_text = ".".join(parts)

        # Loại bỏ chuỗi ingredient lặp lại
        preprocessed_text = re.sub(
            r"(ingredients?\s*)+",
            "Ingredients ",
            preprocessed_text,
            flags=re.IGNORECASE,
        )

        # Phát hiện và xử lý đặc biệt cho danh sách nguyên liệu
        ingredients_pattern = re.search(
            r"ingredients?[:\s]+(.*?)instructions?",
            preprocessed_text,
            re.IGNORECASE | re.DOTALL,
        )
        instructions_pattern = re.search(
            r"instructions?[:\s]+(.*)", preprocessed_text, re.IGNORECASE | re.DOTALL
        )

        # Nếu phát hiện mẫu của công thức nấu ăn
        if ingredients_pattern and instructions_pattern:
            logger.info("Recipe format detected, using special summarization logic")

            ingredients = ingredients_pattern.group(1).strip()
            instructions = instructions_pattern.group(1).strip()

            # Loại bỏ các số đo lường trùng lặp trong nguyên liệu
            ingredients = re.sub(r"(\d+\s*[a-z]+)\s*\1", r"\1", ingredients)

            # Loại bỏ các phần lặp lại trong ingredients
            for common_ingredient in [
                "egg",
                "eggs",
                "cream",
                "sugar",
                "vanilla",
                "honey",
            ]:
                pattern = r"(\b" + common_ingredient + r"\b).*?(\b\1\b)"
                ingredients = re.sub(pattern, r"\1", ingredients, flags=re.IGNORECASE)

            # Tạo một văn bản tóm tắt trực tiếp cho công thức
            clean_ingredients = re.sub(r"[\d,.]+\s*[xX]", "", ingredients)
            clean_ingredients = re.sub(r"[\d,.]+\s*[a-z]+\s+of", "", clean_ingredients)

            # Xử lý lại instructions trước khi tóm tắt
            clean_instructions = instructions

            # Loại bỏ các dòng bắt đầu bằng các lượng nguyên liệu lặp lại
            clean_instructions = re.sub(
                r"\n\s*\d+\s*[a-z]+\s+[a-z]+.*?\n", "\n", clean_instructions
            )

            # Tách instructions thành các bước
            instruction_steps = clean_instructions.split(".")

            # Xử lý từng bước để loại bỏ sự lặp lại
            processed_steps = []
            for step in instruction_steps:
                if len(step.strip()) > 0:
                    # Loại bỏ các từ lặp lại trong mỗi bước
                    for word in ["egg", "eggs", "cream", "oven", "ready", "sugar"]:
                        step = re.sub(
                            r"(\b" + word + r"\b).*?(\b\1\b)",
                            r"\1",
                            step,
                            flags=re.IGNORECASE,
                        )
                    processed_steps.append(step)

            # Ghép lại các bước đã xử lý
            clean_instructions = ". ".join(processed_steps)

            # Tóm tắt hướng dẫn chỉ nếu đủ dài
            if len(clean_instructions.split()) > 50:
                try:
                    summary_instructions = SUMMARIZER(
                        clean_instructions,
                        max_length=adjusted_max_length // 2,
                        min_length=adjusted_min_length // 2,
                        do_sample=False,
                    )[0]["summary_text"]

                    # Kiểm tra chất lượng tóm tắt
                    if _is_poor_quality_summary(
                        summary_instructions
                    ) or _is_same_as_input(summary_instructions, clean_instructions):
                        logger.warning(
                            "Poor quality summary detected for instructions, using fallback summarization"
                        )
                        summary_instructions = _fallback_summarize(clean_instructions)
                except Exception as e:
                    logger.error(f"Error summarizing instructions: {e}")
                    summary_instructions = _fallback_summarize(clean_instructions)
            else:
                summary_instructions = clean_instructions

            # Xử lý thêm tóm tắt để loại bỏ các từ lặp lại
            summary_instructions = _post_process_summary(summary_instructions)

            # Kết hợp lại
            final_summary = f"Ingredients: {clean_ingredients.strip()}\n\nInstructions: {summary_instructions.strip()}"

            # Kiểm tra kết quả tóm tắt so với input
            if _is_same_as_input(final_summary, text):
                logger.warning("Summary same as input, using fallback summarizer")
                return _fallback_summarize(text)

            return final_summary

        # Giảm kích thước nếu văn bản quá dài
        if len(preprocessed_text.split()) > 1024:
            # Lấy 1024 từ đầu tiên
            truncated_text = " ".join(preprocessed_text.split()[:1024])
            logger.info(
                f"Text truncated from {len(preprocessed_text.split())} to 1024 words for summarization"
            )
        else:
            truncated_text = preprocessed_text

        # Thử với temperature và sampling thấp hơn để có kết quả ổn định hơn
        summary = SUMMARIZER(
            truncated_text,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            do_sample=False,  # Tắt sampling
            num_beams=2,  # Giảm beam search
        )[0]["summary_text"]

        if summary and len(summary) > 0:
            summarized_text = (
                summary[0]["summary_text"] if isinstance(summary, list) else summary
            )

            # Kiểm tra chất lượng của bản tóm tắt
            if _is_poor_quality_summary(summarized_text) or _is_same_as_input(
                summarized_text, text
            ):
                logger.warning(
                    "Poor quality summary or same as input detected, using fallback summarizer"
                )
                return _fallback_summarize(text)

            # Xử lý sau tóm tắt để loại bỏ lặp lại
            processed_summary = _post_process_summary(summarized_text)
            return processed_summary

        return _fallback_summarize(text)
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        logger.error(traceback.format_exc())
        return _fallback_summarize(text)


def _is_vietnamese_dialogue(text: str) -> bool:
    """
    Kiểm tra xem văn bản có phải là đối thoại tiếng Việt không

    Args:
        text: Văn bản cần kiểm tra

    Returns:
        True nếu văn bản là đối thoại tiếng Việt, False nếu không
    """
    # Kiểm tra các dấu hiệu của đối thoại tiếng Việt
    import re

    # Các mẫu đặc trưng hơn cho đối thoại trong tiếng Việt
    # Pattern 1: Tìm kiếm định dạng "X:" hoặc "Người X:" ở đầu dòng hoặc đầu văn bản
    dialogue_pattern1 = re.compile(
        r"(?:^|\n)\s*(?:[A-Z][a-z]*:|[A-Z]:|Người \d+:|Nhân viên:|Nam:|Nữ:|Hỏi:|Đáp:|Người hỏi:|Người trả lời:|Nhân vật \d+:|Bạn:|Tôi:|[A-Z]\d+:)",
        re.MULTILINE,
    )

    # Pattern 2: Tìm các cặp tên người đối thoại theo khuôn mẫu A-B, P-Q, H-T, v.v.
    dialogue_pattern2 = re.compile(r"(?:^|\n)\s*(?:A|B|P|Q|H|T):\s", re.MULTILINE)

    # Kiểm tra xem có bất kỳ mẫu đối thoại nào không
    has_dialogue_pattern = bool(
        dialogue_pattern1.search(text) or dialogue_pattern2.search(text)
    )

    # Đếm số lượt đối thoại "X:" trong văn bản
    dialogue_turns = len(re.findall(r"(?:^|\n)\s*\w+:", text, re.MULTILINE))

    # Kiểm tra xem có ít nhất 2 lượt đối thoại không
    has_multiple_turns = dialogue_turns >= 2

    # Kiểm tra các từ tiếng Việt phổ biến để xác định ngôn ngữ
    vietnamese_words = [
        "của",
        "và",
        "các",
        "có",
        "không",
        "được",
        "đã",
        "trong",
        "tôi",
        "chúng",
        "với",
        "cho",
        "mình",
        "cảm ơn",
        "xin chào",
        "vâng",
        "dạ",
        "bạn",
        "chị",
        "anh",
        "này",
        "khi",
        "đang",
        "rất",
    ]
    count = sum(1 for word in vietnamese_words if word.lower() in text.lower())

    # Văn bản được xác định là đối thoại tiếng Việt nếu:
    # 1. Có mẫu đối thoại theo các định dạng thông dụng
    # 2. Có ít nhất 2 lượt đối thoại
    # 3. Có ít nhất 3 từ tiếng Việt phổ biến
    is_vietnamese = count >= 3

    logger.info(
        f"Dialogue detection: pattern={has_dialogue_pattern}, turns={dialogue_turns}, vietnamese words={count}"
    )

    return (has_dialogue_pattern or has_multiple_turns) and is_vietnamese


def _fallback_summarize(text: str) -> str:
    """
    Hàm tóm tắt dự phòng khi mô hình chính thất bại

    Args:
        text: Văn bản cần tóm tắt

    Returns:
        Bản tóm tắt
    """
    logger.info("Using fallback summarization method")

    import re
    from collections import Counter

    # Nếu văn bản quá ngắn, trả về nguyên bản
    if len(text.split()) < 20:
        return text

    # Trường hợp đặc biệt: Nếu là đối thoại tiếng Việt, sử dụng phương pháp trích xuất đối thoại
    dialogue_pattern = re.compile(r"([A-Z]):\s*(.*?)(?=\s*[A-Z]:\s*|$)", re.DOTALL)
    dialogue_matches = dialogue_pattern.findall(text)

    if len(dialogue_matches) >= 2:
        logger.info(f"Detected dialogue format with {len(dialogue_matches)} turns")
        # Tìm các câu hỏi trong đối thoại
        questions = []
        for speaker, content in dialogue_matches:
            if "?" in content:
                questions.append((speaker, content.strip()))

        # Tìm câu trả lời dài nhất
        answers = []
        speakers = set([speaker for speaker, _ in dialogue_matches])
        for speaker in speakers:
            speaker_turns = [content for s, content in dialogue_matches if s == speaker]
            if speaker_turns:
                longest_turn = max(speaker_turns, key=len)
                answers.append((speaker, longest_turn.strip()))

        # Tạo tóm tắt từ đối thoại
        summary_parts = []

        # Thêm các câu hỏi (tối đa 2)
        for speaker, content in questions[:2]:
            summary_parts.append(f"{speaker}: {content}")

        # Thêm câu trả lời quan trọng nhất
        for speaker, content in answers:
            if len(summary_parts) < 3 and not any(
                speaker in part for part in summary_parts
            ):
                summary_parts.append(f"{speaker}: {content}")

        # Thêm lượt cuối cùng nếu chưa có
        last_speaker, last_content = dialogue_matches[-1]
        if not any(last_speaker in part for part in summary_parts):
            summary_parts.append(f"{last_speaker}: {last_content.strip()}")

        dialogue_summary = " ".join(summary_parts)
        logger.info(f"Created dialogue summary with {len(summary_parts)} turns")
        return dialogue_summary

    # Xử lý văn bản thông thường
    # Tách văn bản thành các câu
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    logger.info(
        f"Split text into {len(sentences)} sentences for fallback summarization"
    )

    if len(sentences) <= 3:
        return text

    # Làm sạch các câu
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    # Loại bỏ các từ stopwords phổ biến trong tiếng Việt
    stopwords = {
        "và",
        "của",
        "có",
        "các",
        "là",
        "những",
        "được",
        "trong",
        "đã",
        "cho",
        "này",
        "với",
        "tại",
        "về",
        "như",
        "từ",
        "không",
        "đến",
        "để",
        "theo",
        "một",
        "ra",
        "trên",
        "cũng",
        "khi",
        "nhiều",
        "sẽ",
        "phải",
        "vào",
    }

    # Tính tần suất xuất hiện của từng từ
    words = " ".join(clean_sentences).lower().split()
    word_counts = Counter([w for w in words if w not in stopwords and len(w) > 1])

    # Đánh giá điểm cho mỗi câu dựa trên tần suất từ xuất hiện trong nó
    sentence_scores = []
    for i, sentence in enumerate(clean_sentences):
        score = 0
        for word in sentence.lower().split():
            if word in word_counts:
                score += word_counts[word]

        # Ưu tiên câu đầu và câu cuối
        position_boost = 0
        if i < len(clean_sentences) * 0.2:  # Câu ở đầu văn bản
            position_boost = 0.1
        elif i > len(clean_sentences) * 0.8:  # Câu ở cuối văn bản
            position_boost = 0.05

        # Ưu tiên câu hỏi
        question_boost = 0.15 if "?" in sentence else 0

        final_score = (
            score
            * (1 + position_boost + question_boost)
            / max(1, len(sentence.split()))
        )
        sentence_scores.append((i, sentence, final_score))

    # Sắp xếp câu theo điểm
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)

    # Lấy top 30% câu có điểm cao nhất
    top_count = max(3, int(len(clean_sentences) * 0.3))
    top_sentences = sorted_sentences[:top_count]

    # Sắp xếp lại theo thứ tự xuất hiện ban đầu để giữ mạch văn bản
    ordered_top_sentences = sorted(top_sentences, key=lambda x: x[0])

    # Tạo bản tóm tắt
    summary_parts = []
    last_idx = -1
    for idx, sentence, _ in ordered_top_sentences:
        if last_idx != -1 and idx > last_idx + 1:
            summary_parts.append("...")
        summary_parts.append(sentence)
        last_idx = idx

    summary = " ".join(summary_parts)
    logger.info(f"Fallback summary created with {len(summary_parts)} sentences")

    return summary


def _summarize_vietnamese_dialogue(text: str) -> str:
    """
    Tóm tắt đặc biệt cho đối thoại tiếng Việt

    Args:
        text: Văn bản đối thoại tiếng Việt cần tóm tắt

    Returns:
        Bản tóm tắt đối thoại tiếng Việt
    """
    try:
        import re

        logger.info(f"Starting Vietnamese dialogue summary, text length: {len(text)}")

        # Sử dụng nhiều mẫu regex khác nhau để tăng khả năng phát hiện đối thoại

        # 1. Tách theo mẫu chuẩn của đối thoại "X: nội dung"
        dialogue_pattern1 = re.compile(
            r"([A-Z][a-z]*|[A-Z]|Nam|Nữ|Người \d+|Nhân viên|Hỏi|Đáp|P|Q|H|T):\s*(.*?)(?=\s*(?:[A-Z][a-z]*|[A-Z]|Nam|Nữ|Người \d+|Nhân viên|Hỏi|Đáp|P|Q|H|T):|$)",
            re.DOTALL,
        )
        matches1 = dialogue_pattern1.findall(text)
        turns1 = [
            (speaker, content.strip())
            for speaker, content in matches1
            if content.strip()
        ]

        logger.info(f"Extracted {len(turns1)} dialogue turns from pattern 1")

        # 2. Tách theo ngắt dòng và nhãn người nói
        lines = text.split("\n")
        turns2 = []
        current_speaker = None
        current_content = []

        for line in lines:
            # Tìm mẫu "X:" hoặc "Người X:" ở đầu dòng
            match = re.match(
                r"^([A-Z][a-z]*|[A-Z]|Nam|Nữ|Người \d+|Nhân viên|Hỏi|Đáp|P|Q|H|T):\s*(.*)",
                line.strip(),
            )
            if match:
                # Nếu đã có speaker trước đó, lưu lại turn đó
                if current_speaker and current_content:
                    turns2.append((current_speaker, " ".join(current_content)))
                    current_content = []

                # Bắt đầu một turn mới
                current_speaker = match.group(1)
                if match.group(2):  # Nếu có nội dung trên cùng dòng
                    current_content.append(match.group(2))
            elif (
                line.strip() and current_speaker
            ):  # Tiếp tục turn hiện tại nếu dòng không rỗng
                current_content.append(line.strip())

        # Thêm turn cuối cùng nếu có
        if current_speaker and current_content:
            turns2.append((current_speaker, " ".join(current_content)))

        logger.info(f"Extracted {len(turns2)} dialogue turns from pattern 2")

        # 3. Sử dụng mẫu linh hoạt hơn khi hai cách trên không hiệu quả
        if len(turns1) < 2 and len(turns2) < 2:
            # Mẫu linh hoạt hơn để phát hiện đối thoại
            dialogue_pattern3 = re.compile(
                r"([A-Z][a-zA-Z]*|[A-Z])[\s:]\s*(.*?)(?=\n[A-Z][a-zA-Z]*[\s:]|$)",
                re.DOTALL,
            )
            matches3 = dialogue_pattern3.findall(text)
            turns3 = [
                (speaker, content.strip())
                for speaker, content in matches3
                if content.strip()
            ]
            logger.info(f"Extracted {len(turns3)} dialogue turns from pattern 3")

            # Nếu vẫn chưa tìm được đủ lượt, thử một mẫu cuối cùng - tách theo dòng
            if len(turns3) < 2:
                turns4 = []
                for line in lines:
                    # Cố gắng tìm bất kỳ chuỗi nào bắt đầu bằng chữ cái đầu tiên viết hoa
                    match = re.match(r"^([^:]+):\s*(.*)", line.strip())
                    if match and len(match.group(2).strip()) > 0:
                        speaker = match.group(1).strip()
                        content = match.group(2).strip()
                        turns4.append((speaker, content))
                logger.info(f"Extracted {len(turns4)} dialogue turns from pattern 4")

                if len(turns4) >= 2:
                    turns3 = turns4
        else:
            turns3 = []

        # Chọn kết quả tốt nhất từ các phương pháp
        turns = (
            turns1
            if len(turns1) >= max(len(turns2), len(turns3))
            else (turns2 if len(turns2) >= len(turns3) else turns3)
        )

        # In ra thông tin để debug
        if turns:
            logger.info(f"Selected best pattern with {len(turns)} turns")
            speakers = set(speaker for speaker, _ in turns)
            logger.info(f"Found speakers: {', '.join(speakers)}")

        # Nếu không tìm thấy đối thoại hợp lệ, sử dụng phương pháp tóm tắt dự phòng
        if len(turns) < 2:
            logger.warning(
                f"Not enough dialogue turns detected ({len(turns)}), using fallback summarization"
            )
            return _fallback_summarize(text)

        # Xác định các nhân vật chính trong đối thoại
        speakers = set(speaker for speaker, _ in turns)
        logger.info(f"Identified {len(speakers)} speakers: {', '.join(speakers)}")

        # Tìm các nội dung quan trọng cho mỗi người nói
        important_turns = []
        for speaker in speakers:
            speaker_turns = [content for s, content in turns if s == speaker]
            logger.info(f"Speaker {speaker} has {len(speaker_turns)} turns")

            # Tìm câu hỏi và câu trả lời quan trọng
            question_turns = []
            important_content = []

            for content in speaker_turns:
                if "?" in content:
                    question_turns.append(content)
                elif len(content.split()) > 15:  # Phát biểu dài thường quan trọng
                    important_content.append(content)

            # Thêm câu hỏi quan trọng nhất (nếu có)
            if question_turns:
                important_turns.append(
                    {
                        "speaker": speaker,
                        "content": max(question_turns, key=len),
                        "is_question": True,
                        "length": len(max(question_turns, key=len).split()),
                    }
                )

            # Thêm phát biểu dài nhất (nếu có)
            if important_content:
                longest_content = max(important_content, key=len)

                # Chỉ thêm nếu chưa có turn nào của speaker này
                if not any(turn["speaker"] == speaker for turn in important_turns):
                    important_turns.append(
                        {
                            "speaker": speaker,
                            "content": longest_content,
                            "is_question": False,
                            "length": len(longest_content.split()),
                        }
                    )

        # Nếu không tìm thấy turn quan trọng nào, sử dụng mặc định lấy turn dài nhất cho mỗi người
        if not important_turns:
            for speaker in speakers:
                speaker_turns = [content for s, content in turns if s == speaker]
                if speaker_turns:
                    longest_turn = max(speaker_turns, key=len)
                    important_turns.append(
                        {
                            "speaker": speaker,
                            "content": longest_turn,
                            "is_question": "?" in longest_turn,
                            "length": len(longest_turn.split()),
                        }
                    )

        # Sắp xếp theo trình tự xuất hiện đầu tiên trong đối thoại
        speaker_order = []
        for speaker, _ in turns:
            if speaker not in speaker_order:
                speaker_order.append(speaker)

        # Ưu tiên câu hỏi và sắp xếp theo thứ tự xuất hiện các lượt nói
        def sort_key(turn):
            question_priority = 0 if turn["is_question"] else 1
            appearance_order = (
                speaker_order.index(turn["speaker"])
                if turn["speaker"] in speaker_order
                else len(speaker_order)
            )
            return (question_priority, appearance_order)

        important_turns.sort(key=sort_key)

        # Tạo bản tóm tắt đối thoại
        summary_parts = []
        included_speakers = set()

        # Đảm bảo có ít nhất một lượt của mỗi người nói
        for speaker in speaker_order:
            speaker_turns = [
                turn for turn in important_turns if turn["speaker"] == speaker
            ]
            if speaker_turns and speaker not in included_speakers:
                turn = speaker_turns[0]  # Lấy lượt quan trọng nhất
                summary_parts.append(f"{turn['speaker']}: {turn['content']}")
                included_speakers.add(speaker)
                if len(summary_parts) >= 4:  # Giới hạn số lượng lượt tối đa
                    break

        # Thêm lượt kết thúc nếu chưa được thêm
        last_speaker, last_content = turns[-1]
        if last_speaker not in included_speakers and len(summary_parts) < 4:
            summary_parts.append(f"{last_speaker}: {last_content}")

        # Ghép thành đoạn tóm tắt
        summary = " ".join(summary_parts)

        logger.info(f"Created dialogue summary with {len(summary_parts)} turns")

        # Kiểm tra xem kết quả có giống input không
        if _is_same_as_input(summary, text):
            logger.warning(
                "Vietnamese dialogue summary is too similar to input, using fallback"
            )
            return _fallback_summarize(text)

        return summary
    except Exception as e:
        logger.error(f"Error in Vietnamese dialogue summarization: {str(e)}")
        logger.error(traceback.format_exc())
        return _fallback_summarize(text)


def _is_poor_quality_summary(summary_text: str) -> bool:
    """
    Kiểm tra xem tóm tắt có chất lượng kém không

    Args:
        summary_text: Văn bản tóm tắt cần kiểm tra

    Returns:
        True nếu tóm tắt có chất lượng kém, False nếu không
    """
    import re

    # Kiểm tra độ dài và độ đa dạng từ vựng
    if len(summary_text.split()) < 5 or len(set(summary_text.split())) < 3:
        return True

    # Kiểm tra lặp từ nghiêm trọng
    for word in ["egg", "eggs", "cream", "oven", "ready", "sugar", "g", "ml"]:
        # Nếu từ lặp lại hơn 3 lần
        if summary_text.lower().count(" " + word + " ") > 3:
            return True

    # Kiểm tra mẫu lặp lại
    if re.search(r"(\b\w+\b)(\s+\1\b){2,}", summary_text) or re.search(
        r"(\d+\s*){4,}", summary_text
    ):
        return True

    # Kiểm tra chất lượng dựa trên số lượng câu
    sentences = summary_text.split(".")
    if len(sentences) < 2:
        # Nếu chỉ có 1 câu, kiểm tra xem có ý nghĩa không
        if len(sentences[0].split()) < 5:
            return True

    return False


def _post_process_summary(summary_text: str) -> str:
    """
    Xử lý sau tóm tắt để loại bỏ các vấn đề về chất lượng

    Args:
        summary_text: Văn bản tóm tắt cần xử lý

    Returns:
        Văn bản tóm tắt đã được xử lý
    """
    import re

    # Loại bỏ các từ lặp lại liên tiếp
    processed_text = re.sub(r"(\b\w+\b)(\s+\1\b)+", r"\1", summary_text)

    # Xử lý cho các từ thường lặp lại trong công thức
    for word in ["egg", "eggs", "cream", "oven", "ready", "sugar", "g", "ml"]:
        # Loại bỏ các lặp lại trong cùng câu
        sentences = processed_text.split(".")
        for i, sentence in enumerate(sentences):
            # Nếu từ xuất hiện nhiều hơn 1 lần trong câu
            if sentence.lower().count(" " + word + " ") > 1:
                # Tạo pattern để giữ lại lần xuất hiện đầu tiên và loại bỏ các lần sau
                pattern = r"(.*?\b" + word + r"\b)(.*?\b" + word + r"\b.*)"
                sentences[i] = re.sub(pattern, r"\1", sentence, flags=re.IGNORECASE)
        processed_text = ".".join(sentences)

    return processed_text


def extract_topics_with_zero_shot(
    text: str, candidate_topics: List[str] = None, max_topics: int = 5
) -> List[Dict[str, Any]]:
    """
    Trích xuất các chủ đề sử dụng Zero-shot classification

    Args:
        text: Văn bản cần phân tích chủ đề
        candidate_topics: Danh sách các chủ đề ứng cử viên
        max_topics: Số lượng topic tối đa trả về

    Returns:
        Danh sách các chủ đề đã được phân loại và điểm số liên quan
    """
    if not text:
        return []

    if TOPIC_EXTRACTOR is None:
        logger.warning("Topic extraction model not initialized, initializing now...")
        init_models()

    if TOPIC_EXTRACTOR is None:
        logger.error("Failed to initialize topic extraction model")
        return []

    topics = candidate_topics or COMMON_TOPICS
    logger.info(f"Starting topic extraction with {len(topics)} candidate topics")
    logger.info(f"Input text (first 100 chars): {text[:100]}...")

    try:
        if (
            isinstance(TOPIC_EXTRACTOR, str)
            and TOPIC_EXTRACTOR == "distilbert-base-uncased"
        ):
            # Sử dụng DistilBERT để tính độ tương đồng với topics
            text_embedding = get_embedding_distilbert(text)
            if text_embedding is None:
                logger.error("Failed to get text embedding")
                return []

            logger.info(
                f"Generated text embedding with shape: {text_embedding.shape if hasattr(text_embedding, 'shape') else 'unknown'}"
            )

            # Phát hiện từ khóa chính từ văn bản
            keywords = extract_keywords(text, top_n=8)
            logger.info(f"Extracted keywords: {keywords}")

            # Tạo embedding từ từ khóa
            keyword_text = " ".join(keywords) if keywords else text
            keyword_embedding = get_embedding_distilbert(keyword_text)
            if keyword_embedding is None:
                logger.warning(
                    "Failed to get keyword embedding, using text embedding instead"
                )
                keyword_embedding = text_embedding

            # Chuyển đổi embeddings thành numpy array nếu cần
            if not isinstance(text_embedding, np.ndarray):
                text_embedding = np.array(text_embedding)
            if not isinstance(keyword_embedding, np.ndarray):
                keyword_embedding = np.array(keyword_embedding)

            # Kết hợp nhúng văn bản và nhúng từ khóa với kiểm tra kích thước
            if text_embedding.shape == keyword_embedding.shape:
                combined_embedding = 0.6 * text_embedding + 0.4 * keyword_embedding
            else:
                logger.warning(
                    f"Embedding shapes don't match: text={text_embedding.shape}, keyword={keyword_embedding.shape}"
                )
                combined_embedding = text_embedding

            # Tạo từ điển domain và tính toán domain chính
            domain_keywords = {
                "Computer Science & Technology": [
                    "computer",
                    "software",
                    "programming",
                    "algorithm",
                    "data",
                    "web",
                    "internet",
                    "user interface",
                    "ui",
                    "ux",
                    "app",
                    "application",
                    "code",
                    "database",
                    "network",
                ],
                "Medicine & Health": [
                    "medicine",
                    "drug",
                    "medical",
                    "symptom",
                    "treatment",
                    "disease",
                    "patient",
                    "health",
                    "doctor",
                    "hospital",
                    "pill",
                    "medication",
                    "therapy",
                    "cold",
                    "relief",
                ],
                "Music & Arts": [
                    "music",
                    "song",
                    "visual",
                    "record",
                    "oral",
                    "arts",
                    "representational",
                    "formal",
                    "visual record",
                    "literature",
                    "author",
                    "poet",
                    "novel",
                    "fiction",
                ],
                "History & Culture": [
                    "history",
                    "tradition",
                    "formal",
                    "anthropology",
                    "culture",
                    "society",
                    "social",
                ],
            }

            # Phát hiện lĩnh vực chính dựa trên từ khóa
            domain_scores = {}
            text_lower = text.lower()
            for domain, domain_kw_list in domain_keywords.items():
                score = 0
                for kw in domain_kw_list:
                    if kw in text_lower:
                        score += 1
                domain_scores[domain] = score

            logger.info(f"Domain scores: {domain_scores}")

            # Xác định domain chính
            if domain_scores:
                primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            else:
                primary_domain = None
            logger.info(f"Primary domain: {primary_domain}")

            # Ánh xạ từ topic sang domain
            topic_domain_map = {
                "Computer Science": "Computer Science & Technology",
                "Technology": "Computer Science & Technology",
                "Engineering": "Computer Science & Technology",
                "Medicine": "Medicine & Health",
                "Health": "Medicine & Health",
                "Music": "Music & Arts",
                "Art": "Music & Arts",
                "Literature": "Music & Arts",
                "Entertainment": "Music & Arts",
                "History": "History & Culture",
                "Anthropology": "History & Culture",
                "Sociology": "History & Culture",
            }

            results = []
            all_similarity_scores = {}

            for topic in topics:
                # Tạo embedding cho topic
                topic_embedding = get_embedding_distilbert(topic)
                if topic_embedding is None:
                    logger.warning(
                        f"Failed to get embedding for topic '{topic}', skipping"
                    )
                    continue

                # Chuyển đổi topic embedding thành numpy array nếu cần
                if not isinstance(topic_embedding, np.ndarray):
                    topic_embedding = np.array(topic_embedding)

                # Kiểm tra kích thước embedding
                if topic_embedding.shape != combined_embedding.shape:
                    logger.warning(
                        f"Topic embedding shape {topic_embedding.shape} doesn't match combined embedding shape {combined_embedding.shape}, skipping"
                    )
                    continue

                # Tính độ tương đồng cosine
                norm_combined = np.linalg.norm(combined_embedding)
                norm_topic = np.linalg.norm(topic_embedding)

                if norm_combined == 0 or norm_topic == 0:
                    similarity = 0
                else:
                    similarity = np.dot(combined_embedding, topic_embedding) / (
                        norm_combined * norm_topic
                    )

                # 1. Tăng điểm cho topic có từ khóa trực tiếp
                direct_keyword_match = False
                for kw in keywords:
                    if kw.lower() in topic.lower() or topic.lower() in kw.lower():
                        similarity += 0.2  # Tăng từ 0.15 lên 0.2
                        direct_keyword_match = True
                        break

                # 2. Tăng điểm nếu topic thuộc domain chính
                if primary_domain and topic in topic_domain_map:
                    if topic_domain_map[topic] == primary_domain:
                        similarity += 0.1

                # 3. Tăng cường tổng thể độ tương đồng để cải thiện điểm DistilBERT
                similarity = similarity * 1.3  # Tăng tổng thể điểm số lên 30%

                # Đảm bảo chỉ số là kiểu float
                similarity = float(similarity)
                all_similarity_scores[topic] = similarity

                results.append(
                    {
                        "topic": topic,
                        "score": similarity,
                        "method": "distilbert-cosine-similarity",
                    }
                )

            # Log các điểm số trước khi lọc
            top_scores = sorted(
                all_similarity_scores.items(), key=lambda x: x[1], reverse=True
            )[:5]
            logger.info(f"Top 5 topics before filtering: {top_scores}")

            # Sắp xếp theo điểm số giảm dần và lọc ngưỡng
            results = sorted(results, key=lambda x: x["score"], reverse=True)

            # Giảm ngưỡng từ 0.25 xuống 0.15 để có nhiều chủ đề DistilBERT hơn
            threshold = 0.15  # Đặt ngưỡng thấp hơn để bao gồm nhiều topic DistilBERT
            filtered_results = [r for r in results if r["score"] > threshold][
                :max_topics
            ]
            logger.info(
                f"Filtered results with threshold {threshold}: {filtered_results}"
            )

            return filtered_results
        else:
            # Zero-shot classification
            logger.info("Using zero-shot classification pipeline")
            result = TOPIC_EXTRACTOR(text, topics, multi_label=True)

            # Log kết quả
            logger.info(
                f"Zero-shot results: {result['labels'][:5]} with scores {result['scores'][:5]}"
            )

            # Thay đổi ngưỡng từ 0.0 lên 0.25
            threshold = 0.15  # Đặt ngưỡng phù hợp để lọc các topic có điểm số thấp
            logger.info(f"Using threshold: {threshold}")

            # Tăng cường điểm cho zero-shot để cạnh tranh tốt hơn với LDA
            formatted_results = [
                {
                    "topic": label,
                    "score": float(score) * 1.2,  # Tăng điểm 20% để cạnh tranh với LDA
                    "method": "zero-shot-classification",
                }
                for label, score in zip(result["labels"], result["scores"])
                if score > threshold
            ]

            logger.info(f"Formatted results count: {len(formatted_results)}")
            return formatted_results[:max_topics]
    except Exception as e:
        logger.error(f"Error in topic extraction with zero-shot: {e}")
        logger.error(traceback.format_exc())
        return []


def extract_topics_with_lda(text: str, num_topics: int = 5) -> List[Dict[str, Any]]:
    """
    Trích xuất các chủ đề sử dụng LDA (Latent Dirichlet Allocation)

    Args:
        text: Văn bản cần phân tích chủ đề
        num_topics: Số lượng chủ đề cần trích xuất

    Returns:
        Danh sách các chủ đề và từ khóa liên quan
    """
    if not text or NLP is None:
        return []

    try:
        # Tiền xử lý văn bản
        doc = NLP(text)

        # Lọc ra các danh từ, động từ và tính từ quan trọng
        filtered_tokens = [
            token.lemma_.lower()
            for token in doc
            if (
                token.is_alpha
                and not token.is_stop
                and token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]
                and len(token.text) > 2
            )
        ]

        filtered_text = " ".join(filtered_tokens)

        # Nếu không đủ token, giảm yêu cầu xuống 5 tokens (thay vì 10)
        if len(filtered_tokens) < 5:
            # Thử dùng tất cả tokens có ý nghĩa
            filtered_tokens = [
                token.lemma_.lower()
                for token in doc
                if (token.is_alpha and len(token.text) > 2)
            ]
            filtered_text = " ".join(filtered_tokens)

            # Nếu vẫn không đủ, tạo một topic giả với score cao
            if len(filtered_tokens) < 5:
                logger.warning(
                    f"Not enough tokens for LDA: {len(filtered_tokens)} tokens"
                )
                # Trả về ít nhất một chủ đề mặc định với từ khóa từ văn bản
                top_words = [token.text for token in doc if token.is_alpha][:5]
                if not top_words:
                    top_words = ["text", "document"]

                return [
                    {
                        "topic_id": "lda_topic_fallback",
                        "topic": f"{top_words[0].title()} Analysis",
                        "keywords": top_words[:5],
                        "score": 0.65,  # Đặt điểm đủ cao để vượt ngưỡng
                        "method": "lda",
                    }
                ]

        # Vectorize văn bản
        vectorizer = CountVectorizer(max_features=1000)
        X = vectorizer.fit_transform([filtered_text])

        # Áp dụng LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics, random_state=42, max_iter=10
        )
        lda.fit(X)

        # Lấy các từ khóa cho mỗi chủ đề
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(lda.components_):
            # Lấy 10 từ khóa hàng đầu cho chủ đề
            top_features_idx = topic.argsort()[: -10 - 1 : -1]
            top_features = [feature_names[i] for i in top_features_idx]

            # Tính topic coherence và tăng giá trị score để đảm bảo vượt qua ngưỡng
            raw_coherence = np.mean(topic[top_features_idx]) / np.sum(lda.components_)

            # Nhân với hệ số để tăng score, đảm bảo LDA topics đủ score để hiển thị
            # Nhân với 15 thay vì 40 để giảm sự ưu tiên của LDA
            enhanced_coherence = float(raw_coherence * 15)

            # Đảm bảo score trong khoảng hợp lý (0.3 - 0.75) - giảm ngưỡng tối thiểu
            enhanced_coherence = min(max(enhanced_coherence, 0.3), 0.75)

            # Tạo tên topic từ 2 từ khóa đầu tiên
            topic_name = f"{top_features[0].title()} {top_features[1].title()}"

            # Log để dễ theo dõi
            logger.info(
                f"LDA topic {topic_idx}: '{topic_name}', raw coherence: {raw_coherence}, enhanced: {enhanced_coherence}"
            )

            topics.append(
                {
                    "topic_id": f"lda_topic_{topic_idx}",
                    "topic": topic_name,  # Thêm trường topic
                    "keywords": top_features,
                    "score": enhanced_coherence,  # Sử dụng giá trị score đã tăng cường
                    "method": "lda",
                }
            )

        # Sắp xếp theo điểm số
        topics = sorted(topics, key=lambda x: x["score"], reverse=True)

        # Đảm bảo luôn trả về ít nhất một topic
        if not topics:
            top_words = [token.text for token in doc if token.is_alpha][:5]
            if not top_words:
                top_words = ["text", "document"]

            topics = [
                {
                    "topic_id": "lda_topic_fallback",
                    "topic": f"{top_words[0].title()} Analysis",
                    "keywords": top_words[:5],
                    "score": 0.65,
                    "method": "lda",
                }
            ]

        return topics
    except Exception as e:
        logger.error(f"Error in topic extraction with LDA: {e}")
        logger.error(traceback.format_exc())
        return []


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Trích xuất các từ khóa quan trọng từ văn bản sử dụng spaCy

    Args:
        text: Văn bản cần trích xuất từ khóa
        top_n: Số lượng từ khóa cần trả về

    Returns:
        Danh sách các từ khóa quan trọng
    """
    if not text or NLP is None:
        return []

    try:
        doc = NLP(text)

        # Tìm các cụm danh từ và tên riêng
        noun_chunks = [
            chunk.text.lower()
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) <= 3
        ]
        named_entities = [ent.text.lower() for ent in doc.ents]

        # Tìm các từ quan trọng (noun, verb, adj)
        important_tokens = [
            token.lemma_.lower()
            for token in doc
            if (
                token.is_alpha
                and not token.is_stop
                and token.pos_ in ["NOUN", "PROPN"]
                and len(token.text) > 2
            )
        ]

        # Kết hợp tất cả từ khóa và loại bỏ trùng lặp
        all_keywords = noun_chunks + named_entities + important_tokens

        # Tính tần suất xuất hiện
        keyword_freq = {}
        for kw in all_keywords:
            if kw in keyword_freq:
                keyword_freq[kw] += 1
            else:
                keyword_freq[kw] = 1

        # Sắp xếp theo tần suất
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)

        # Lấy các từ khóa có tần suất cao nhất
        top_keywords = [kw for kw, freq in sorted_keywords[:top_n]]

        return top_keywords
    except Exception as e:
        logger.error(f"Error in keyword extraction: {e}")
        return []


def process_visible_content(
    visible_content: str, max_topics: int = 5
) -> Dict[str, Any]:
    """
    Xử lý nội dung hiển thị để tóm tắt và trích xuất chủ đề

    Args:
        visible_content: Nội dung hiển thị của trang web
        max_topics: Số lượng topic tối đa trả về

    Returns:
        Dict chứa bản tóm tắt, chủ đề và từ khóa đã trích xuất
    """
    if not visible_content:
        return {"summary": "", "topics": [], "keywords": []}

    # Kiểm tra xem models đã được khởi tạo chưa
    if SUMMARIZER is None or TOPIC_EXTRACTOR is None or NLP is None:
        init_models()

    try:
        # 1. Tóm tắt nội dung
        summary = summarize_text(visible_content)

        # 2. Trích xuất chủ đề sử dụng Zero-shot classification
        topics_zero_shot = extract_topics_with_zero_shot(summary, max_topics=max_topics)

        # 3. Trích xuất chủ đề sử dụng LDA
        topics_lda = extract_topics_with_lda(visible_content)

        # Log để kiểm tra
        logger.info(f"LDA topics extracted: {len(topics_lda)}")
        if topics_lda:
            logger.info(f"First LDA topic: {topics_lda[0]}")

        # 4. Trích xuất từ khóa
        keywords = extract_keywords(summary)

        # Tạo topics từ keywords với score thấp hơn LDA
        keyword_topics = []
        for kw in keywords[:5]:  # Sử dụng 5 từ khóa hàng đầu
            keyword_topics.append(
                {
                    "topic": kw.title(),
                    "score": 0.6,  # Đặt điểm thấp hơn LDA một chút
                    "method": "keyword-extraction",
                }
            )

        # Theo dõi số lượng topic từ mỗi phương pháp
        method_topics = {
            "distilbert": topics_zero_shot,
            "lda": topics_lda,
            "keyword": keyword_topics,
        }

        # Đảm bảo mỗi phương pháp có ít nhất một đại diện
        final_topics = []

        # Chia đều số lượng topics cho mỗi phương pháp
        topics_per_method = max(1, max_topics // 3)

        # 1. Thêm topics từ LDA
        if topics_lda:
            final_topics.extend(topics_lda[:topics_per_method])
            logger.info(f"Added {len(topics_lda[:topics_per_method])} LDA topics")

        # 2. Thêm topics từ DistilBERT
        if topics_zero_shot:
            final_topics.extend(topics_zero_shot[:topics_per_method])
            logger.info(
                f"Added {len(topics_zero_shot[:topics_per_method])} DistilBERT topics"
            )

        # 3. Thêm topics từ keyword extraction
        if keyword_topics:
            final_topics.extend(keyword_topics[:topics_per_method])
            logger.info(
                f"Added {len(keyword_topics[:topics_per_method])} Keyword topics"
            )

        # Nếu chưa đủ số lượng, thêm topics từ các phương pháp còn dư
        remaining_slots = max_topics - len(final_topics)
        if remaining_slots > 0:
            # Gom tất cả topics còn lại
            remaining_topics = []
            if len(topics_lda) > topics_per_method:
                remaining_topics.extend(topics_lda[topics_per_method:])
            if len(topics_zero_shot) > topics_per_method:
                remaining_topics.extend(topics_zero_shot[topics_per_method:])
            if len(keyword_topics) > topics_per_method:
                remaining_topics.extend(keyword_topics[topics_per_method:])

            # Sắp xếp theo score và thêm vào
            remaining_topics.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_topics.extend(remaining_topics[:remaining_slots])

            logger.info(
                f"Added {min(remaining_slots, len(remaining_topics))} additional topics to fill remaining slots"
            )

        # Loại bỏ các chủ đề trùng lặp
        seen_topics = set()
        unique_topics = []

        for topic in final_topics:
            topic_name = topic.get("topic", "")
            if topic_name and topic_name.lower() not in seen_topics:
                seen_topics.add(topic_name.lower())
                unique_topics.append(topic)

        # Nếu vẫn thiếu, thử thêm từ các phương pháp còn thiếu
        if len(unique_topics) < max_topics:
            for method, topics in method_topics.items():
                if not any(t.get("method") == method for t in unique_topics) and topics:
                    best_topic = sorted(
                        topics, key=lambda x: x.get("score", 0), reverse=True
                    )[0]
                    if best_topic.get("topic", "").lower() not in seen_topics:
                        unique_topics.append(best_topic)
                        seen_topics.add(best_topic.get("topic", "").lower())
                        logger.info(
                            f"Added {method} topic to ensure representation: {best_topic}"
                        )

                    if len(unique_topics) >= max_topics:
                        break

        # Giới hạn số lượng topic tối đa và sắp xếp theo score
        final_topics = sorted(
            unique_topics[:max_topics], key=lambda x: x.get("score", 0), reverse=True
        )

        # Log phân phối cuối cùng của các topic
        method_count = {
            "LDA": sum(1 for t in final_topics if t.get("method") == "lda"),
            "DistilBERT": sum(
                1
                for t in final_topics
                if t.get("method")
                in ["distilbert-cosine-similarity", "zero-shot-classification"]
            ),
            "Keyword": sum(
                1 for t in final_topics if t.get("method") == "keyword-extraction"
            ),
        }
        logger.info(
            f"Final topics distribution: LDA: {method_count['LDA']}, DistilBERT: {method_count['DistilBERT']}, Keyword: {method_count['Keyword']}"
        )

        return {
            "summary": summary,
            "topics": final_topics,
            "keywords": keywords,
        }
    except Exception as e:
        logger.error(f"Error in process_visible_content: {e}")
        logger.error(traceback.format_exc())
        return {"summary": "", "topics": [], "keywords": []}


def extract_dbpedia_topics(visible_content: str) -> List[str]:
    """
    Trích xuất tên các topic DBpedia từ nội dung hiển thị

    Args:
        visible_content: Nội dung hiển thị của trang web

    Returns:
        Danh sách tên các topic liên quan tới DBpedia
    """
    result = process_visible_content(visible_content)

    # Lấy tên các topic từ kết quả phân tích
    topic_names = []

    # Thêm các topic từ zero-shot classification
    for topic_item in result["topics"]:
        if "topic" in topic_item:
            topic_names.append(topic_item["topic"])
        elif "topic_id" in topic_item and "keywords" in topic_item:
            # Thêm từ khóa đầu tiên từ LDA topics
            if topic_item["keywords"]:
                topic_names.append(topic_item["keywords"][0].title())

    # Thêm một số từ khóa quan trọng như là topics
    if result["keywords"]:
        for keyword in result["keywords"][:3]:  # Chỉ lấy 3 từ khóa hàng đầu
            topic_names.append(keyword.title())

    # Loại bỏ trùng lặp và giữ nguyên thứ tự
    unique_topics = []
    for topic in topic_names:
        if topic not in unique_topics:
            unique_topics.append(topic)

    return unique_topics


def batch_process_entries(
    entries: List[Dict[str, Any]], max_topics_per_entry: int = 5
) -> List[Dict[str, Any]]:
    """
    Xử lý hàng loạt các entry để tóm tắt và trích xuất chủ đề

    Args:
        entries: Danh sách các entry (mỗi entry có visible_content)
        max_topics_per_entry: Số lượng topic tối đa cho mỗi entry

    Returns:
        Danh sách các entry đã được xử lý với summary và topics
    """
    if not entries:
        return []

    # Kiểm tra xem models đã được khởi tạo chưa
    if SUMMARIZER is None or TOPIC_EXTRACTOR is None or NLP is None:
        init_models()

    processed_entries = []

    for entry in entries:
        if "visible_content" not in entry or not entry["visible_content"]:
            processed_entries.append(entry)
            continue

        try:
            # Xử lý nội dung và thêm thông tin tóm tắt và topics
            result = process_visible_content(
                entry["visible_content"], max_topics=max_topics_per_entry
            )

            # Cập nhật entry với thông tin mới
            updated_entry = entry.copy()
            updated_entry["summary"] = result["summary"]

            # Tạo danh sách các topic name giới hạn số lượng
            topic_names = [
                topic["topic"] for topic in result["topics"] if "topic" in topic
            ][:max_topics_per_entry]

            # Thêm từ các LDA topics nếu cần
            lda_topics = []
            for topic in result["topics"]:
                if "topic_id" in topic and "keywords" in topic and topic["keywords"]:
                    lda_topics.append(topic["keywords"][0].title())

            # Loại bỏ trùng lặp và giữ nguyên thứ tự, đồng thời giới hạn số lượng
            all_topics = []
            for topic in topic_names + lda_topics:
                if topic not in all_topics:
                    all_topics.append(topic)
                    if len(all_topics) >= max_topics_per_entry:
                        break

            updated_entry["ai_topics"] = all_topics
            updated_entry["ai_keywords"] = result["keywords"]

            processed_entries.append(updated_entry)
        except Exception as e:
            logger.error(f"Error processing entry: {e}")
            processed_entries.append(entry)

    return processed_entries


def _is_same_as_input(summary: str, original_text: str) -> bool:
    """
    Kiểm tra xem bản tóm tắt có giống với văn bản gốc không

    Args:
        summary: Văn bản tóm tắt
        original_text: Văn bản gốc

    Returns:
        True nếu tóm tắt giống văn bản gốc, False nếu không
    """

    # Loại bỏ khoảng trắng, dấu câu để so sánh nội dung thuần
    def normalize_text(text):
        # Loại bỏ dấu câu và khoảng trắng
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Nếu tóm tắt quá ngắn so với nguyên bản (dưới 15%), coi như không phải tóm tắt
    if len(summary) < len(original_text) * 0.15:
        return False

    # Chuẩn hóa cả hai văn bản
    norm_summary = normalize_text(summary)
    norm_original = normalize_text(original_text)

    # Nếu tóm tắt quá dài (trên 85% văn bản gốc), kiểm tra độ tương đồng
    if len(norm_summary) > len(norm_original) * 0.85:
        # Tính tỷ lệ trùng khớp
        if norm_summary == norm_original:
            return True

        # Kiểm tra xem tóm tắt có chứa 90% văn bản gốc không
        if norm_original in norm_summary or norm_summary in norm_original:
            return True

    # Kiểm tra các câu trùng lặp
    summary_sentences = re.split(r"[.!?]", norm_summary)
    original_sentences = re.split(r"[.!?]", norm_original)

    # Đếm số câu trùng khớp
    matching_sentences = 0
    for s_sentence in summary_sentences:
        s_sentence = s_sentence.strip()
        if not s_sentence:
            continue

        for o_sentence in original_sentences:
            o_sentence = o_sentence.strip()
            if not o_sentence:
                continue

            # Kiểm tra xem câu có giống nhau không
            if (
                s_sentence == o_sentence
                or s_sentence in o_sentence
                or o_sentence in s_sentence
            ):
                matching_sentences += 1
                break

    # Nếu trên 70% câu trùng khớp, coi như tóm tắt không tốt
    if matching_sentences / max(1, len(summary_sentences)) > 0.7:
        return True

    return False


def process_visible_content_khanh(
    visible_content: str, max_topics: int = 5
) -> Dict[str, Any]:
    """
    Phiên bản cải tiến của process_visible_content, ưu tiên LDA và keyword extraction
    hơn so với distilbert-cosine-similarity

    Args:
        visible_content: Nội dung hiển thị của trang web
        max_topics: Số lượng topic tối đa trả về

    Returns:
        Dict chứa bản tóm tắt, chủ đề và từ khóa đã trích xuất
    """
    if not visible_content:
        return {"summary": "", "topics": [], "keywords": []}

    # Kiểm tra xem models đã được khởi tạo chưa
    if SUMMARIZER is None or TOPIC_EXTRACTOR is None or NLP is None:
        init_models()

    try:
        # 1. Tóm tắt nội dung
        summary = summarize_text(visible_content)

        # 2. Trích xuất chủ đề sử dụng Zero-shot classification
        topics_zero_shot = extract_topics_with_zero_shot(summary, max_topics=max_topics)

        # 3. Trích xuất chủ đề sử dụng LDA
        topics_lda = extract_topics_with_lda(visible_content)

        # Log để kiểm tra
        logger.info(f"LDA topics extracted: {len(topics_lda)}")
        if topics_lda:
            logger.info(f"First LDA topic: {topics_lda[0]}")

        # 4. Trích xuất từ khóa
        keywords = extract_keywords(summary)

        # Tạo topics từ keywords với score cao hơn distilbert
        keyword_topics = []
        for idx, kw in enumerate(keywords[:5]):  # Sử dụng 5 từ khóa hàng đầu
            keyword_topics.append(
                {
                    "topic": kw.title(),
                    # Tăng điểm số cho keywords để cao hơn distilbert
                    "score": 0.75 - (idx * 0.05),  # Bắt đầu từ 0.75 thay vì 0.7
                    "method": "keyword-extraction",
                }
            )

        # Kiểm tra nếu nội dung có liên quan đến CS/tech
        tech_keywords = [
            "software",
            "programming",
            "code",
            "algorithm",
            "computer",
            "database",
            "web",
            "app",
            "application",
            "interface",
            "ui",
            "ux",
            "internet",
            "developer",
            "development",
            "website",
            "tech",
            "technology",
            "digital",
            "javascript",
            "python",
            "java",
            "html",
            "css",
            "framework",
        ]

        # Đếm số từ khóa tech trong nội dung
        content_lower = visible_content.lower()
        tech_keyword_count = sum(1 for kw in tech_keywords if kw in content_lower)

        # THAY ĐỔI: Giảm score của zero-shot classification (distilbert) mạnh hơn
        for topic in topics_zero_shot:
            # Giảm điểm số xuống 50% thay vì 20%
            topic["score"] = topic["score"] * 0.5

            # Kiểm tra và giảm điểm thêm cho chủ đề Computer Science nếu nội dung không liên quan
            if (
                topic["topic"] == "Computer Science"
                or topic["topic"] == "Technology"
                or "tech" in topic["topic"].lower()
                or "computer" in topic["topic"].lower()
            ):
                if tech_keyword_count < 2:  # Nếu có ít từ khóa tech trong nội dung
                    # Giảm điểm thêm 50% nữa
                    topic["score"] = topic["score"] * 0.5
                    logger.info(
                        f"Reducing Computer Science related topic score due to irrelevance: {topic['topic']}"
                    )

        # Theo dõi số lượng topic từ mỗi phương pháp
        method_topics = {
            "distilbert": topics_zero_shot,
            "lda": topics_lda,
            "keyword": keyword_topics,
        }

        # Đảm bảo mỗi phương pháp có ít nhất một đại diện
        final_topics = []

        # THAY ĐỔI: Ưu tiên LDA và keyword extraction hơn distilbert
        # Phân phối lại số lượng topics cho mỗi phương pháp
        # LDA: 50%, keyword: 40%, distilbert: 10%
        topics_per_method = {
            "lda": max(1, int(max_topics * 0.5)),
            "keyword": max(1, int(max_topics * 0.4)),
            "distilbert": 1,  # Giới hạn chỉ 1 topic từ distilbert
        }

        # 1. Thêm topics từ LDA (ưu tiên cao nhất)
        if topics_lda:
            lda_limit = topics_per_method["lda"]
            final_topics.extend(topics_lda[:lda_limit])
            logger.info(f"Added {len(topics_lda[:lda_limit])} LDA topics")

        # 2. Thêm topics từ keyword extraction (ưu tiên thứ hai)
        if keyword_topics:
            keyword_limit = topics_per_method["keyword"]
            final_topics.extend(keyword_topics[:keyword_limit])
            logger.info(f"Added {len(keyword_topics[:keyword_limit])} Keyword topics")

        # 3. Thêm topics từ DistilBERT (ưu tiên thấp nhất)
        if topics_zero_shot:
            # Sắp xếp theo điểm số để lấy topic distilbert tốt nhất
            best_distilbert_topics = sorted(
                topics_zero_shot, key=lambda x: x.get("score", 0), reverse=True
            )

            # Lọc bỏ các chủ đề Computer Science không liên quan
            filtered_distilbert = []
            for topic in best_distilbert_topics:
                is_cs_topic = (
                    topic["topic"] == "Computer Science"
                    or topic["topic"] == "Technology"
                    or "tech" in topic["topic"].lower()
                    or "computer" in topic["topic"].lower()
                )

                if not is_cs_topic or tech_keyword_count >= 2:
                    filtered_distilbert.append(topic)

            # Nếu không có topic nào sau khi lọc, sử dụng các topic ban đầu
            if not filtered_distilbert:
                filtered_distilbert = best_distilbert_topics

            distilbert_limit = topics_per_method["distilbert"]
            final_topics.extend(filtered_distilbert[:distilbert_limit])
            logger.info(
                f"Added {len(filtered_distilbert[:distilbert_limit])} DistilBERT topics"
            )

        # Nếu chưa đủ số lượng, thêm topics từ các phương pháp LDA và keyword còn dư
        # THAY ĐỔI: Chỉ bổ sung từ LDA và keyword, bỏ qua distilbert
        remaining_slots = max_topics - len(final_topics)
        if remaining_slots > 0:
            # Gom topics từ LDA và keyword còn dư
            remaining_topics = []
            if len(topics_lda) > topics_per_method["lda"]:
                remaining_topics.extend(topics_lda[topics_per_method["lda"] :])
            if len(keyword_topics) > topics_per_method["keyword"]:
                remaining_topics.extend(keyword_topics[topics_per_method["keyword"] :])

            # Sắp xếp theo score và thêm vào
            remaining_topics.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_topics.extend(remaining_topics[:remaining_slots])

            logger.info(
                f"Added {min(remaining_slots, len(remaining_topics))} additional topics to fill remaining slots"
            )

        # Loại bỏ các chủ đề trùng lặp
        seen_topics = set()
        unique_topics = []

        for topic in final_topics:
            topic_name = topic.get("topic", "")
            if topic_name and topic_name.lower() not in seen_topics:
                seen_topics.add(topic_name.lower())
                unique_topics.append(topic)

        # Giới hạn số lượng topic tối đa và sắp xếp theo score
        final_topics = sorted(
            unique_topics[:max_topics], key=lambda x: x.get("score", 0), reverse=True
        )

        # Log phân phối cuối cùng của các topic
        method_count = {
            "LDA": sum(1 for t in final_topics if t.get("method") == "lda"),
            "DistilBERT": sum(
                1
                for t in final_topics
                if t.get("method")
                in ["distilbert-cosine-similarity", "zero-shot-classification"]
            ),
            "Keyword": sum(
                1 for t in final_topics if t.get("method") == "keyword-extraction"
            ),
        }
        logger.info(
            f"Final topics distribution: LDA: {method_count['LDA']}, DistilBERT: {method_count['DistilBERT']}, Keyword: {method_count['Keyword']}"
        )

        return {
            "summary": summary,
            "topics": final_topics,
            "keywords": keywords,
        }
    except Exception as e:
        logger.error(f"Error in process_visible_content_khanh: {e}")
        logger.error(traceback.format_exc())
        return {"summary": "", "topics": [], "keywords": []}
