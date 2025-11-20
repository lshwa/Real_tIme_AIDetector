import os
import tkinter as tk
from tkinter import ttk, font
import joblib
import numpy as np
import kss

from feature_extractor_local import calculate_features, get_feature_names

# --- Constants ---
DEBOUNCE_DELAY_MS = 1000       # 키 입력 후 분석까지 딜레이 (ms)
MIN_CHARS_FOR_DETECTION = 50   # 이 길이 미만이면 분석 안 함


class AiDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- 1. 모델 로드 ---
        self.title("KatFishNet 기반 쉼표-피처 AI 글쓰기 탐지기")
        self.geometry("1200x720")
        self.minsize(950, 650)

        try:
            script_dir = os.path.dirname(__file__)
            # KatFishNet에서 제공한 pipeline (StandardScaler + LogisticRegression)
            self.model = joblib.load(os.path.join(script_dir, "model.joblib"))
        except FileNotFoundError as e:
            self.show_error_and_quit(
                f"필수 모델 파일(model.joblib)을 찾을 수 없습니다: {e.filename}\n"
                "KatFishNet에서 제공한 model.joblib 을 이 폴더에 복사했는지 확인하세요."
            )
            return
        except Exception as e:
            self.show_error_and_quit(f"프로그램 초기화 중 오류 발생:\n{e}")
            return

        # 피처 이름 (쉼표 기반 5개)
        self.feature_names = get_feature_names()

        # 로지스틱 회귀 계수 (판단 이유용)
        try:
            self.coefs = self.model.named_steps["clf"].coef_[0]  # shape (5,)
        except Exception:
            self.coefs = None

        self.debounce_timer = None

        # 확률 smoothing용 히스토리
        self.prob_history = []
        self.HISTORY_LEN = 5

        # --- 2. UI 설정 ---
        self.configure_styles()
        self.create_widgets()

    # ========== 공통 유틸 ==========

    def show_error_and_quit(self, message):
        error_win = tk.Toplevel(self)
        error_win.title("오류")
        tk.Label(error_win, text=message, padx=20, pady=20, justify="left").pack()
        tk.Button(error_win, text="종료", command=self.destroy).pack(pady=10)
        self.withdraw()

    def configure_styles(self):
        # 기본 폰트
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="AppleGothic", size=11)

        self.title_font = font.Font(family="AppleGothic", size=16, weight="bold")
        self.subtitle_font = font.Font(family="AppleGothic", size=10)
        self.gauge_font = font.Font(family="Arial", size=38, weight="bold")
        self.reason_font = font.Font(family="AppleGothic", size=11)
        self.reason_title_font = font.Font(family="AppleGothic", size=12, weight="bold")
        self.status_font = font.Font(family="AppleGothic", size=12, weight="bold")

        s = ttk.Style()
        s.theme_use("default")
        s.configure("TFrame", background="#f8f9fa")
        s.configure("Left.TFrame", background="white")
        s.configure("Right.TFrame", background="#f8f9fa")
        s.configure("Header.TFrame", background="#343a40")
        s.configure("Header.TLabel", background="#343a40", foreground="white")
        s.configure("Gauge.TLabel", background="#f8f9fa")
        s.configure("Status.TLabel", background="#f8f9fa")

    # ========== UI 구성 ==========

    def create_widgets(self):
        root_frame = ttk.Frame(self, style="TFrame")
        root_frame.pack(fill=tk.BOTH, expand=True)

        # 상단 헤더
        header = ttk.Frame(root_frame, style="Header.TFrame", padding=(20, 10))
        header.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(
            header,
            text="KatFishNet | 쉼표 기반 HEI Detector",
            style="Header.TLabel",
            font=self.title_font,
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="쉼표 위치/빈도 + 품사 패턴 5개 피처를 기반으로 AI 글쓰기 의심도를 추정합니다.",
            style="Header.TLabel",
            font=self.subtitle_font,
        ).pack(anchor="w", pady=(4, 0))

        # 좌우 분할
        main_frame = ttk.Frame(root_frame, style="TFrame", padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # 왼쪽: 텍스트 입력
        left_pane = ttk.Frame(paned_window, style="Left.TFrame", padding=(10, 10))
        paned_window.add(left_pane, weight=3)

        ttk.Label(
            left_pane,
            text="✏️ 분석할 글을 입력하세요",
            font=self.reason_title_font,
            background="white",
        ).pack(anchor="w", pady=(0, 5))

        self.text_input = tk.Text(
            left_pane,
            wrap=tk.WORD,
            font=self.default_font,
            relief=tk.FLAT,
            borderwidth=0,
            undo=True,
            background="white",
            highlightthickness=1,
            highlightbackground="#dee2e6",
            highlightcolor="#339af0",
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)

        # 오른쪽: 결과
        right_pane = ttk.Frame(paned_window, style="Right.TFrame", padding=(20, 10))
        paned_window.add(right_pane, weight=2)

        ttk.Label(
            right_pane,
            text="분석 결과",
            font=self.title_font,
            background="#f8f9fa",
        ).pack(pady=(0, 10), anchor="w")

        gauge_frame = ttk.Frame(right_pane, style="Right.TFrame")
        gauge_frame.pack(fill=tk.X, pady=(0, 15))

        self.gauge_label = ttk.Label(
            gauge_frame,
            text="0%",
            style="Gauge.TLabel",
            font=self.gauge_font,
            foreground="#007bff",
        )
        self.gauge_label.pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = ttk.Label(
            gauge_frame,
            text="대기 중",
            style="Status.TLabel",
            font=self.status_font,
            foreground="#6c757d",
        )
        self.status_label.pack(side=tk.LEFT, anchor="s")

        ttk.Separator(right_pane, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        explanation_frame = ttk.Frame(right_pane, style="Right.TFrame")
        explanation_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # 판단 이유
        ttk.Label(
            explanation_frame,
            text="판단 이유",
            font=self.reason_title_font,
            background="#f8f9fa",
        ).pack(anchor="w")
        self.reasons_text = tk.Text(
            explanation_frame,
            font=self.reason_font,
            wrap=tk.WORD,
            height=6,
            relief=tk.FLAT,
            borderwidth=0,
            background="white",
            state=tk.DISABLED,
            foreground="#495057",
            highlightthickness=1,
            highlightbackground="#dee2e6",
        )
        self.reasons_text.pack(fill=tk.X, expand=False, pady=(5, 15))

        # 의심 문장
        ttk.Label(
            explanation_frame,
            text="주요 의심 문장 (가장 AI스러운 문장)",
            font=self.reason_title_font,
            background="#f8f9fa",
        ).pack(anchor="w")
        self.suspect_text = tk.Text(
            explanation_frame,
            font=self.reason_font,
            wrap=tk.WORD,
            height=6,
            relief=tk.FLAT,
            borderwidth=0,
            background="#fffbe6",
            state=tk.DISABLED,
            foreground="#495057",
            highlightthickness=1,
            highlightbackground="#ffe066",
        )
        self.suspect_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # 이벤트 바인딩
        self.text_input.bind("<KeyRelease>", self.on_key_release)

    # ========== 이벤트 & 분석 로직 ==========

    def on_key_release(self, event=None):
        if self.debounce_timer:
            self.after_cancel(self.debounce_timer)
        self.debounce_timer = self.after(DEBOUNCE_DELAY_MS, self.analyze_text)

    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END)
        stripped = text.strip()

        if len(stripped) < MIN_CHARS_FOR_DETECTION:
            self.update_ui(
                0.0, [f"분석을 위해 최소 {MIN_CHARS_FOR_DETECTION}자 이상 입력해주세요."], ""
            )
            return

        # 1) 전체 글 기준 쉼표-피처 (5차원)
        feats_doc = calculate_features(stripped)          # shape (5,)
        doc_prob = float(self.model.predict_proba(feats_doc.reshape(1, -1))[0][1])

        # 2) 문장별 확률
        sentences = kss.split_sentences(stripped)
        sent_probs = []
        for s in sentences:
            if len(s.strip()) < 10:
                continue
            f_s = calculate_features(s)
            p_s = float(self.model.predict_proba(f_s.reshape(1, -1))[0][1])
            sent_probs.append((s, p_s))

        suspect_sentence = ""
        max_sent_prob = 0.0
        sent_probs_only = []

        if sent_probs:
            suspect_sentence, max_sent_prob = max(sent_probs, key=lambda x: x[1])
            sent_probs_only = [p for (_, p) in sent_probs]

        # 3) 전체/문장/분포를 같이 반영해서 combined_prob 계산
        if sent_probs_only:
            avg_sent_prob = float(np.mean(sent_probs_only))
            high_frac = float(np.mean([p > 0.8 for p in sent_probs_only]))  # AI스러운 문장 비율
            low_frac  = float(np.mean([p < 0.3 for p in sent_probs_only]))  # 사람스러운 문장 비율

            # (1) 기본값: 전체(doc)와 문장 평균의 중간
            base = 0.5 * doc_prob + 0.5 * avg_sent_prob

            # (2) low 문장이 많으면 패널티, high 문장이 많으면 약간 보너스
            #     - low_frac = 1 이면 40%까지 깎음
            #     - high_frac = 1 이면 20%까지 올려줌
            penalty = 1.0 - 0.4 * low_frac    # [0.6, 1.0]
            bonus   = 1.0 + 0.2 * high_frac   # [1.0, 1.2]

            combined_prob = base * penalty * bonus

            # [0, 1] 범위로 클리핑
            combined_prob = max(0.0, min(1.0, combined_prob))
        else:
            # 문장이 거의 없으면 그냥 doc_prob만 사용
            combined_prob = float(doc_prob)

        # 4) 시간 방향 smoothing (최근 HISTORY_LEN 번 평균)
        self.prob_history.append(combined_prob)
        window = self.prob_history[-self.HISTORY_LEN:]
        smoothed_prob = float(np.mean(window))

        # 디버그 출력
        print(
            f"[DEBUG] doc_prob={doc_prob:.4f}, "
            f"max_sent_prob={max_sent_prob:.4f}, "
            f"combined={combined_prob:.4f}, "
            f"smoothed={smoothed_prob:.4f}"
        )

        # 5) 판단 이유 생성
        reasons = []
        if smoothed_prob > 0.55 and self.coefs is not None:
            reasons = self.get_analysis_reasons(feats_doc, self.coefs)

        self.update_ui(smoothed_prob, reasons, suspect_sentence)

    # ========== 설명 텍스트 생성 ==========

    def get_analysis_reasons(self, features, coefs):
        """
        KatFishNet 로지스틱 계수(coefs)와 쉼표-피처(features)를 곱해서
        AI일 확률을 높이는 데 기여한 상위 2개 특성을 텍스트로 설명.
        """
        feature_impacts = features * coefs  # 길이 5

        top_indices = np.argsort(feature_impacts)[-2:][::-1]
        reasons = []

        feature_map = {
            "comma_include_sentence_rate_per_text": "문장 중 쉼표를 포함한 문장의 비율이 높습니다.",
            "avg_comma_usage_rate_per_text": "문장 당 쉼표 사용 빈도가 높습니다.",
            "avg_relative_position_per_text": "쉼표가 문장 내 특정 위치(중후반)에 몰려 있습니다.",
            "avg_segment_length_per_text": "쉼표로 나뉜 구간 길이가 일정한 편입니다.",
            "avg_pos_patterns_diversity_score_per_text": "쉼표 앞뒤 품사 패턴이 다양하지 않고 일정합니다.",
        }

        for idx in top_indices:
            if feature_impacts[idx] > 0:
                fname = self.feature_names[idx]
                reasons.append(
                    f"・ {feature_map.get(fname, 'AI 생성물의 통계적 패턴과 유사합니다.')}"
                )

        if not reasons:
            reasons.append("・ 전반적인 쉼표 사용 패턴이 AI 생성 글과 유사한 통계를 보입니다.")

        return reasons

    # ========== UI 업데이트 ==========

    def update_ui(self, probability, reasons, suspect_sentence):
        raw_p = probability
        prob_percent = round(raw_p * 100)

        # 색상: 파랑(0) → 빨강(1) 보간
        r = int(0 + (220 - 0) * raw_p)
        g = int(123 + (53 - 123) * raw_p)
        b = int(255 + (69 - 255) * raw_p)
        color = f"#{r:02x}{g:02x}{b:02x}"

        self.gauge_label.config(text=f"{prob_percent}%", foreground=color)

        # 상태 문구
        if raw_p < 0.35:
            status_text = "인간 작성 가능성이 높습니다."
            status_color = "#2b8a3e"
        elif raw_p < 0.65:
            status_text = "인간/AI 혼합 또는 애매한 영역입니다."
            status_color = "#f08c00"
        else:
            status_text = "AI 작성 의심 구간입니다."
            status_color = "#c92a2a"

        self.status_label.config(text=status_text, foreground=status_color)

        # 판단 이유
        self.reasons_text.config(state=tk.NORMAL)
        self.reasons_text.delete("1.0", tk.END)
        if reasons:
            self.reasons_text.insert(tk.END, "\n".join(reasons))
        else:
            self.reasons_text.insert(
                tk.END, "인간이 작성한 글의 특성이 강하게 나타납니다."
            )
        self.reasons_text.config(state=tk.DISABLED)

        # 의심 문장
        self.suspect_text.config(state=tk.NORMAL)
        self.suspect_text.delete("1.0", tk.END)
        self.suspect_text.insert(tk.END, suspect_sentence)
        self.suspect_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    app = AiDetectorApp()
    app.mainloop()