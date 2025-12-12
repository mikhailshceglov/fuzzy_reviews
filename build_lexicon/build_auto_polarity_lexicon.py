
import argparse
import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set

try:
    import pymorphy3 as pymorphy2  
except ImportError:
    import pymorphy2
from razdel import sentenize, tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------- Russian stopwords (леммы) -----------------
RUSSIAN_STOPWORDS: Set[str] = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "ее", "мне", "есть", "если", "или", "ни",
    "когда", "быть", "кто", "сам", "до", "только", "уже", "для",
    "мы", "тут", "от", "вот", "этот", "который", "наш", "такой", "тот",
    "их", "при", "без", "над", "под", "после", "про", "через",
    "между", "из", "о", "об", "там", "сюда", "туда", "здесь",
    "где", "зачем", "почему", "всегда", "никогда", "никто",
    "ничто", "нигде", "никуда", "поэтому", "впрочем", "ли",
    "же", "ещё", "тоже", "также",
}

# Части речи, которые мы игнорируем (служебные слова) по pymorphy2
IGNORED_POS = {
    "PREP",  # предлог
    "CONJ",  # союз
    "PRCL",  # частица
    "INTJ",  # междометие
}


class SentimentLexiconBuilder:
    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny-sentiment-balanced",
        batch_size: int = 32,
        max_length: int = 256,
        min_freq: int = 20,
        near_zero_eps: float = 0.05,
        near_zero_max_freq: int = 50,
        encoding: str = "utf-8",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_freq = min_freq
        self.near_zero_eps = near_zero_eps
        self.near_zero_max_freq = near_zero_max_freq
        self.encoding = encoding

        # Морфология
        self.morph = pymorphy2.MorphAnalyzer()

        # Устройство для модели
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        print(f"[INFO] Loading sentiment model '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Предвычислим веса меток в [-1, 1]
        num_labels = self.model.config.num_labels
        id2label = self.model.config.id2label
        label_weights = self._build_label_weights(id2label)
        self.label_weights = torch.tensor(
            label_weights, dtype=torch.float32, device=self.device
        )
        print(f"[INFO] Label mapping (id -> weight): {list(enumerate(label_weights))}")

    @staticmethod
    def _build_label_weights(id2label: Dict[int, str]) -> List[float]:
        """
        Построить веса в [-1, 1] для меток сентимента.

        Если удаётся распознать 'neg'/'pos'/'neu', используем их.
        Иначе равномерно растягиваем метки по [-1, 1].
        """
        num_labels = len(id2label)
        weights = [0.0] * num_labels

        has_neg = has_pos = has_neu = False
        for i, lab in id2label.items():
            l = lab.lower()
            if "neg" in l or "нег" in l or "negative" in l:
                weights[i] = -1.0
                has_neg = True
            elif "pos" in l or "поз" in l or "positive" in l:
                weights[i] = 1.0
                has_pos = True
            elif "neu" in l or "нейтр" in l or "neutral" in l:
                weights[i] = 0.0
                has_neu = True

        # Если хоть что-то найдено, оставляем как есть
        if has_neg or has_pos or has_neu:
            return weights

        # Фоллбек: равномерно от -1 до 1
        if num_labels == 1:
            return [0.0]
        for i in range(num_labels):
            weights[i] = -1.0 + 2.0 * i / (num_labels - 1)
        return weights

    def _lemmatize_sentence(self, text: str) -> Set[str]:
        """
        Лемматизировать предложение и вернуть множество лемм (уникальные в пределах предложения).
        """
        lemmas: Set[str] = set()
        for token in tokenize(text):
            raw = token.text.lower()
            if not raw.isalpha():
                continue
            parsed = self.morph.parse(raw)[0]
            if parsed.tag.POS in IGNORED_POS:
                continue
            lemma = parsed.normal_form
            if lemma in RUSSIAN_STOPWORDS:
                continue
            lemmas.add(lemma)
        return lemmas

    def _iter_sentences_from_file(self, path: Path) -> Iterable[Tuple[str, Set[str]]]:
        """
        Итерироваться по предложениям файла, возвращая (текст, множество лемм).
        """
        with path.open("r", encoding=self.encoding, errors="ignore") as f:
            text = f.read()

        for s in sentenize(text):
            sent_text = s.text.strip()
            if not sent_text:
                continue
            lemmas = self._lemmatize_sentence(sent_text)
            if not lemmas:
                continue
            yield sent_text, lemmas

    @staticmethod
    def _iter_corpus_files(input_paths: List[Path]) -> Iterable[Path]:
        """
        Обойти все текстовые файлы.
        Если путь — директория, рекурсивно ищем *.txt.
        """
        for p in input_paths:
            if p.is_dir():
                for sub in p.rglob("*.txt"):
                    if sub.is_file():
                        yield sub
            elif p.is_file():
                yield p

    def _sentiment_scores_for_batch(self, sentences: List[str]) -> List[float]:
        """
        Посчитать сентимент в [-1, 1] для батча предложений.
        """
        if not sentences:
            return []

        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits  # [batch, num_labels]
            probs = torch.softmax(logits, dim=-1)
            # sentiment_score = sum(p_i * weight_i)
            scores = torch.matmul(probs, self.label_weights)
        return scores.cpu().tolist()

    def build_lexicon(self, inputs: List[Path]) -> Dict[str, Dict[str, float]]:
        """
        Главный метод: пройти по корпусу, посчитать частоты и средние полярности.
        """
        lemma_freq: Dict[str, int] = defaultdict(int)
        lemma_score_sum: Dict[str, float] = defaultdict(float)

        batch_sentences: List[str] = []
        batch_lemmas: List[Set[str]] = []

        total_sentences = 0
        total_files = 0

        for path in self._iter_corpus_files(inputs):
            total_files += 1
            print(f"[INFO] Processing file: {path}")
            for sent_text, lemmas in self._iter_sentences_from_file(path):
                batch_sentences.append(sent_text)
                batch_lemmas.append(lemmas)

                if len(batch_sentences) >= self.batch_size:
                    scores = self._sentiment_scores_for_batch(batch_sentences)
                    for sent_lemmas, score in zip(batch_lemmas, scores):
                        # freq = число предложений, где лемма встретилась
                        for lemma in sent_lemmas:
                            lemma_freq[lemma] += 1
                            lemma_score_sum[lemma] += float(score)
                    total_sentences += len(batch_sentences)
                    print(
                        f"[INFO] Processed {total_sentences} sentences...",
                        end="\r",
                    )
                    batch_sentences.clear()
                    batch_lemmas.clear()

        # Хвост батча
        if batch_sentences:
            scores = self._sentiment_scores_for_batch(batch_sentences)
            for sent_lemmas, score in zip(batch_lemmas, scores):
                for lemma in sent_lemmas:
                    lemma_freq[lemma] += 1
                    lemma_score_sum[lemma] += float(score)
            total_sentences += len(batch_sentences)
            batch_sentences.clear()
            batch_lemmas.clear()

        print(
            f"\n[INFO] Finished. Total files: {total_files}, "
            f"total sentences: {total_sentences}"
        )
        print(f"[INFO] Unique lemmas before filtering: {len(lemma_freq)}")

        lexicon: Dict[str, Dict[str, float]] = {}

        for lemma, freq in lemma_freq.items():
            if freq < self.min_freq:
                continue
            avg_score = lemma_score_sum[lemma] / freq
            # Фильтрация слабых и редких
            if abs(avg_score) < self.near_zero_eps and freq < self.near_zero_max_freq:
                continue
            lexicon[lemma] = {
                "polarity": round(avg_score, 4),
                "freq": int(freq),
            }

        print(f"[INFO] Lemmas after filtering: {len(lexicon)}")
        return lexicon

    @staticmethod
    def save_json(lexicon: Dict[str, Dict[str, float]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved JSON lexicon to {path}")

    @staticmethod
    def save_csv(
        lexicon: Dict[str, Dict[str, float]],
        path: Path,
        sort_by: str = "freq",  # или "polarity_abs"
    ) -> None:
        items = [
            (lemma, data["polarity"], data["freq"])
            for lemma, data in lexicon.items()
        ]
        if sort_by == "polarity_abs":
            items.sort(key=lambda x: abs(x[1]), reverse=True)
        else:  # freq
            items.sort(key=lambda x: x[2], reverse=True)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lemma", "polarity", "freq"])
            for lemma, polarity, freq in items:
                writer.writerow([lemma, f"{polarity:.4f}", freq])
        print(f"[INFO] Saved CSV lexicon to {path} (sorted by {sort_by})")

    @staticmethod
    def _print_sample(lexicon: Dict[str, Dict[str, float]], top_n: int = 20) -> None:
        """
        Мини-валидация: вывести топы лемм.
        """
        if not lexicon:
            print("[WARN] Lexicon is empty, nothing to show.")
            return

        items = [
            (lemma, data["polarity"], data["freq"])
            for lemma, data in lexicon.items()
        ]
        # Top positive
        top_pos = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
        # Top negative
        top_neg = sorted(items, key=lambda x: x[1])[:top_n]
        # Near zero
        near_zero = sorted(items, key=lambda x: abs(x[1]))[:top_n]

        def _print_block(title: str, rows: List[Tuple[str, float, int]]) -> None:
            print(f"\n{title}:")
            print("-" * 40)
            for lemma, polarity, freq in rows:
                print(f"{lemma:20s} polarity={polarity:7.4f}  freq={freq}")

        _print_block("Top positive lemmas", top_pos)
        _print_block("Top negative lemmas", top_neg)
        _print_block("Near-zero lemmas", near_zero)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build auto-polarity lexicon for Russian reviews.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input text files or directories with *.txt files.",
    )
    parser.add_argument(
        "--model-name",
        default="cointegrated/rubert-tiny-sentiment-balanced",
        help="HuggingFace model name for sentence sentiment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for sentiment model inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for the model.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=20,
        help="Minimum number of sentences a lemma must appear in.",
    )
    parser.add_argument(
        "--near-zero-eps",
        type=float,
        default=0.05,
        help="Threshold for |polarity| considered 'near zero'.",
    )
    parser.add_argument(
        "--near-zero-max-freq",
        type=int,
        default=50,
        help="Max freq for near-zero lemmas to be filtered out.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding of input text files.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="lexicon.json",
        help="Path to save JSON lexicon.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="lexicon.csv",
        help="Path to save CSV lexicon.",
    )
    parser.add_argument(
        "--csv-sort-by",
        choices=["freq", "polarity_abs"],
        default="freq",
        help="Sorting key for CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]

    builder = SentimentLexiconBuilder(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_freq=args.min_freq,
        near_zero_eps=args.near_zero_eps,
        near_zero_max_freq=args.near_zero_max_freq,
        encoding=args.encoding,
    )

    lexicon = builder.build_lexicon(input_paths)

    builder.save_json(lexicon, Path(args.json_out))
    builder.save_csv(lexicon, Path(args.csv_out), sort_by=args.csv_sort_by)

    # Минимальная валидация — выводим топы
    builder._print_sample(lexicon, top_n=20)


if __name__ == "__main__":
    main()
