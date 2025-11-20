import numpy as np

# ============================================================
#  KatFishNet 쉼표 기반 특성 추출기 통합 버전
#  - 원래 comma_feature_analysis.py + pos_tagging.py 내용 축약
#  - gui_app.py 에서 바로 import 해서 사용
# ============================================================

# ===== 1. konlpy 기반 POS 태거 (Kkma → 실패 시 Okt) =====

_pos_func = None
try:
    from konlpy.tag import Kkma
    _kkma = Kkma()

    def _pos_func(sent):
        # [(morph, tag), ...]
        return _kkma.pos(sent)

except Exception:
    from konlpy.tag import Okt
    _okt = Okt()

    def _pos_func(sent):
        # Okt는 pos만 제공. (morph, tag) 형식으로 맞춤
        return _okt.pos(sent, norm=True, stem=True)


# ===== 2. 쉼표 기반 분석 함수 (원래 analyze_comma_usage) =====

def analyze_comma_usage(sentences, morphs, pos):
    results = {}

    total_comma_count_per_text = 0
    sentence_with_comma_count_per_text = 0
    sentence_without_comma_count_per_text = 0
    total_sentence_count_per_text = len(sentences)
    total_morph_count_per_text = 0

    for sent, morp in zip(sentences, morphs):
        comma_count_in_sentence = morp.count(',')
        total_comma_count_per_text += comma_count_in_sentence
        total_morph_count_per_text += len(morp)

        if comma_count_in_sentence > 0:
            sentence_with_comma_count_per_text += 1
        else:
            sentence_without_comma_count_per_text += 1

    if total_sentence_count_per_text > 0:
        comma_include_sentence_rate_per_text = (
            sentence_with_comma_count_per_text / total_sentence_count_per_text
        )
    else:
        comma_include_sentence_rate_per_text = 0.0

    if total_morph_count_per_text > 0:
        avg_comma_usage_rate_per_text = (
            total_comma_count_per_text / total_morph_count_per_text
        )
    else:
        avg_comma_usage_rate_per_text = 0.0

    relative_positions_per_sentence = []
    segment_lengths_per_sentence = []
    pos_patterns_per_sentence = []
    pos_patterns_diversity_score_per_sentence = []
    comma_usage_rate_per_sentence = []
    avg_relative_position_per_sentence = []
    std_relative_position_per_sentence = []
    avg_segment_length_per_sentence = []
    std_segment_length_per_sentence = []

    for morp, p in zip(morphs, pos):
        if ',' in morp:
            num_commas = morp.count(',')

            # 쉼표 위치(상대 위치)
            comma_positions = [i for i, x in enumerate(morp) if x == ',']
            relative_positions = [pos_i / len(morp) for pos_i in comma_positions]
            relative_positions_per_sentence.append(relative_positions)

            # 쉼표로 나뉜 segment 길이
            segment_lengths = []
            prev_idx = 0
            for idx in comma_positions:
                segment_lengths.append(idx - prev_idx)
                prev_idx = idx + 1
            segment_lengths.append(len(morp) - prev_idx)
            segment_lengths_per_sentence.append(segment_lengths)

            # 쉼표 앞뒤 품사 패턴
            pos_patterns = []
            for idx in comma_positions:
                if idx > 0 and idx < len(p) - 1:
                    prev_pos = p[idx - 1]
                    next_pos = p[idx + 1]
                    pos_patterns.append((prev_pos, next_pos))
            pos_patterns_per_sentence.append(pos_patterns)

            # 품사 패턴 다양성 점수
            if len(pos_patterns) == 0:
                pos_patterns_diversity_score = 0.0
            else:
                pos_patterns_diversity_score = len(set(pos_patterns)) / len(pos_patterns)
            pos_patterns_diversity_score_per_sentence.append(
                pos_patterns_diversity_score
            )

            num_morphs = len(morp)
            comma_usage_rate_per_sentence.append(num_commas / num_morphs)

            avg_relative_position = float(np.mean(relative_positions))
            std_relative_position = float(np.std(relative_positions))
            avg_segment_length = float(np.mean(segment_lengths))
            std_segment_length = float(np.std(segment_lengths))

            avg_relative_position_per_sentence.append(avg_relative_position)
            std_relative_position_per_sentence.append(std_relative_position)
            avg_segment_length_per_sentence.append(avg_segment_length)
            std_segment_length_per_sentence.append(std_segment_length)

    # 문장 레벨 → 텍스트 레벨 집계
    if avg_relative_position_per_sentence:
        avg_relative_position_per_text = float(
            np.mean(avg_relative_position_per_sentence)
        )
        std_relative_position_per_text = float(
            np.std(avg_relative_position_per_sentence)
        )
    else:
        avg_relative_position_per_text = 0.0
        std_relative_position_per_text = 0.0

    if avg_segment_length_per_sentence:
        avg_segment_length_per_text = float(
            np.mean(avg_segment_length_per_sentence)
        )
        std_segment_length_per_text = float(
            np.std(avg_segment_length_per_sentence)
        )
    else:
        avg_segment_length_per_text = 0.0
        std_segment_length_per_text = 0.0

    if pos_patterns_diversity_score_per_sentence:
        avg_pos_patterns_diversity_score_per_text = float(
            np.mean(pos_patterns_diversity_score_per_sentence)
        )
        std_pos_patterns_diversity_score_per_text = float(
            np.std(pos_patterns_diversity_score_per_sentence)
        )
    else:
        avg_pos_patterns_diversity_score_per_text = 0.0
        std_pos_patterns_diversity_score_per_text = 0.0

    if comma_usage_rate_per_sentence:
        avg_comma_usage_rate_per_text = float(
            np.mean(comma_usage_rate_per_sentence)
        )
        std_comma_usage_rate_per_text = float(
            np.std(comma_usage_rate_per_sentence)
        )
    else:
        avg_comma_usage_rate_per_text = 0.0
        std_comma_usage_rate_per_text = 0.0

    results["total_comma_count_per_text"] = total_comma_count_per_text
    results["sentence_with_comma_count_per_text"] = sentence_with_comma_count_per_text
    results["sentence_without_comma_count_per_text"] = (
        sentence_without_comma_count_per_text
    )
    results["total_sentence_count_per_text"] = total_sentence_count_per_text
    results["total_morph_count_per_text"] = total_morph_count_per_text

    results["comma_include_sentence_rate_per_text"] = (
        comma_include_sentence_rate_per_text
    )
    results["avg_comma_usage_rate_per_text"] = avg_comma_usage_rate_per_text
    results["avg_relative_position_per_text"] = avg_relative_position_per_text
    results["std_relative_position_per_text"] = std_relative_position_per_text
    results["avg_segment_length_per_text"] = avg_segment_length_per_text
    results["std_segment_length_per_text"] = std_segment_length_per_text
    results["avg_pos_patterns_diversity_score_per_text"] = (
        avg_pos_patterns_diversity_score_per_text
    )
    results["std_pos_patterns_diversity_score_per_text"] = (
        std_pos_patterns_diversity_score_per_text
    )
    results["std_comma_usage_rate_per_text"] = std_comma_usage_rate_per_text

    return results


# ===== 3. KatFishNet 학습에 썼던 5개 피처 이름 =====

FEATURE_KEYS = [
    "comma_include_sentence_rate_per_text",
    "avg_comma_usage_rate_per_text",
    "avg_relative_position_per_text",
    "avg_segment_length_per_text",
    "avg_pos_patterns_diversity_score_per_text",
]


# ===== 4. 텍스트 1개 → 쉼표 기반 5개 피처 =====

def extract_comma_features(text: str):
    """
    원문 텍스트 하나를 받아서 KatFishNet 훈련 때 쓰던 5개 쉼표-기반 피처를 반환.
    반환: list[float] 길이 5 (FEATURE_KEYS 순서)
    """
    if not text or not text.strip():
        return [0.0] * len(FEATURE_KEYS)

    # 1) 문장 나누기 (kss)
    try:
        import kss
        sentences = list(kss.split_sentences(text))
    except Exception:
        sentences = [text]

    # 2) 각 문장에 대해 형태소/품사 태깅
    morphs_per_sent = []
    pos_per_sent = []
    for s in sentences:
        pairs = _pos_func(s)  # [(morph, tag), ...]
        morphs = [w for (w, t) in pairs]
        tags = [t for (w, t) in pairs]
        morphs_per_sent.append(morphs)
        pos_per_sent.append(tags)

    # 3) 쉼표 기반 통계 계산
    comma_res = analyze_comma_usage(sentences, morphs_per_sent, pos_per_sent)

    feats = [
        comma_res.get("comma_include_sentence_rate_per_text", 0.0),
        comma_res.get("avg_comma_usage_rate_per_text", 0.0),
        comma_res.get("avg_relative_position_per_text", 0.0),
        comma_res.get("avg_segment_length_per_text", 0.0),
        comma_res.get("avg_pos_patterns_diversity_score_per_text", 0.0),
    ]

    # NaN 방지
    feats = [0.0 if (isinstance(v, float) and (v != v)) else v for v in feats]
    return feats


# ============================================================
#  gui_app.py 에서 사용할 공용 인터페이스
# ============================================================

def calculate_features(text: str) -> np.ndarray:
    """
    KatFishNet 쉼표 기반 5개 피처를 numpy 배열로 반환.
    model.joblib (Pipeline: StandardScaler + LogisticRegression)의 입력과 동일.
    """
    feats = extract_comma_features(text)
    feats = np.asarray(feats, dtype=float)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def get_feature_names():
    """훈련에 사용된 피처 이름 리스트."""
    return list(FEATURE_KEYS)