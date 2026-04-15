#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set

# ---- IMPORTANT: choose an interactive backend BEFORE importing pyplot ----
def force_interactive_backend(preferred: str = "qt"):
    import matplotlib
    current = matplotlib.get_backend().lower()
    if any(x in current for x in ["qt", "tk", "wx", "gtk", "macosx"]):
        return

    tried = []
    if preferred.lower().startswith("q"):
        for b in ["QtAgg", "Qt5Agg", "qtagg"]:
            try:
                matplotlib.use(b, force=True)
                return
            except Exception as e:
                tried.append((b, str(e)))
    for b in ["TkAgg", "tkagg"]:
        try:
            matplotlib.use(b, force=True)
            return
        except Exception as e:
            tried.append((b, str(e)))

    raise RuntimeError(
        "Could not load an interactive Matplotlib backend (Qt/Tk).\n"
        "If you're on a headless server, use X forwarding or run locally.\n"
        f"Tried: {tried}"
    )

force_interactive_backend()

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from functools import lru_cache


# -----------------------------
# Data model
# -----------------------------
@dataclass
class QAItem:
    image_id: str
    rgb_path: str
    thermal_path: str
    category: str
    question_id: str
    question: str
    options: List[str]
    answer: str  # model answer
    applicability_score: Optional[float]
    temp_summary: Optional[Dict[str, Any]]
    source_json: str
    row_idx: int  # index inside that source_json list (critical for later patching)


def normalize_sample_pct(p: float) -> float:
    return p * 100.0 if p <= 1.0 else p


def _as_list_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    raise ValueError("JSON must be a list of QA records (or a dict with a 'results' list).")


def parse_records_from_raw(raw: Any, source_json: str) -> List[QAItem]:
    records = _as_list_records(raw)
    items: List[QAItem] = []
    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        items.append(
            QAItem(
                image_id=str(r.get("image_id", "")).strip(),
                rgb_path=str(r.get("rgb_path", "")).strip(),
                thermal_path=str(r.get("thermal_path", "")).strip(),
                category=str(r.get("category", "")).strip(),
                question_id=str(r.get("question_id", "")).strip(),
                question=str(r.get("question", "")).strip(),
                options=list(r.get("options", [])) if isinstance(r.get("options", []), list) else [],
                answer=str(r.get("answer", "")).strip(),
                applicability_score=(float(r["applicability_score"]) if r.get("applicability_score") is not None else None),
                temp_summary=(r.get("temp_summary") if isinstance(r.get("temp_summary"), dict) else None),
                source_json=source_json,
                row_idx=int(idx),
            )
        )
    return items


def load_items(json_path: str) -> List[QAItem]:
    with open(json_path, "r") as f:
        raw = json.load(f)
    return parse_records_from_raw(raw, source_json=os.path.basename(json_path))


def load_items_from_dir(json_dir: str) -> List[QAItem]:
    paths = sorted(
        [
            os.path.join(json_dir, fn)
            for fn in os.listdir(json_dir)
            if fn.lower().endswith(".json")
        ]
    )
    if not paths:
        raise FileNotFoundError(f"No .json files found in directory: {json_dir}")

    all_items: List[QAItem] = []
    for p in paths:
        try:
            all_items.extend(load_items(p))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return all_items


def group_by_image(items: List[QAItem]):
    groups: Dict[Tuple[str, str, str], List[QAItem]] = defaultdict(list)
    order: List[Tuple[str, str, str]] = []
    seen = set()

    for it in items:
        key = (it.image_id, it.rgb_path, it.thermal_path) if it.image_id else ("", it.rgb_path, it.thermal_path)
        groups[key].append(it)
        if key not in seen:
            order.append(key)
            seen.add(key)

    for _, lst in groups.items():
        lst.sort(key=lambda x: (x.category, x.question_id, x.row_idx))
    return order, groups


def _safe_exists(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def norm_text(s: str) -> str:
    """
    Normalize answers for comparison:
    - strip, lowercase
    - collapse whitespace
    - remove surrounding punctuation
    """
    s = (s or "")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n.;:!?,\"'()[]{}")
    return s


# -----------------------------
# Image loader (stable)
# -----------------------------
class ImageLoader:
    def __init__(self, max_dim: int = 900, cache_size: int = 32):
        self.max_dim = max_dim

        @lru_cache(maxsize=cache_size)
        def _load(path: str):
            if not _safe_exists(path):
                return None
            try:
                img = Image.open(path)
                img = img.convert("RGB")
                if self.max_dim and max(img.size) > self.max_dim:
                    img.thumbnail((self.max_dim, self.max_dim), Image.Resampling.LANCZOS)
                return img
            except Exception:
                return None

        self._load = _load

    def get(self, path: str):
        return self._load(path)


# -----------------------------
# Keys + logging
# -----------------------------
LABEL_CORRECT = "correct"
LABEL_CORRECTED = "corrected"  # model incorrect; human picked different option


def make_image_key(image_id: str, rgb_path: str, thermal_path: str) -> str:
    if image_id:
        return f"image_id:{image_id}|rgb:{rgb_path}|thr:{thermal_path}"
    return f"rgb:{rgb_path}|thr:{thermal_path}"


def make_qa_key(qa: QAItem) -> str:
    qid = qa.question_id if qa.question_id else qa.question[:80]
    imgk = make_image_key(qa.image_id, qa.rgb_path, qa.thermal_path)
    return f"{qa.source_json}|row:{qa.row_idx}|{imgk}|qid:{qid}"


class EvalLogger:
    """
    Writes:
      - actions.jsonl (append-only events)
      - checkpoint.json (latest decision per qa_key) for resume
      - report.txt (summary stats)
    """
    def __init__(self, out_dir: str, run_name: str, checkpoint_in: Optional[str] = None):
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name

        self.actions_jsonl = os.path.join(out_dir, f"{run_name}_{ts}_actions.jsonl")
        self.checkpoint_json = os.path.join(out_dir, f"{run_name}_{ts}_checkpoint.json")
        self.report_txt = os.path.join(out_dir, f"{run_name}_{ts}_REPORT.txt")

        self.latest: Dict[str, Dict[str, Any]] = {}
        self.dirty: bool = False

        if checkpoint_in:
            self.load_checkpoint(checkpoint_in)

        self._append_action({"type": "header", "run_name": run_name, "timestamp": ts, "resume_from": checkpoint_in})
        self._write_checkpoint()

    def _append_action(self, obj: Dict[str, Any]):
        with open(self.actions_jsonl, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _write_checkpoint(self):
        tmp = self.checkpoint_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "type": "checkpoint",
                    "run_name": self.run_name,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "items": self.latest,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.checkpoint_json)

    def load_checkpoint(self, path: str):
        try:
            with open(path, "r") as f:
                ck = json.load(f)
            items = ck.get("items", {})
            if isinstance(items, dict):
                self.latest = items
            print(f"[RESUME] Loaded checkpoint with {len(self.latest)} evaluated QA items: {path}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint '{path}': {e}")

    def mark_clean(self):
        self.dirty = False

    def is_done(self, qa: QAItem) -> bool:
        return make_qa_key(qa) in self.latest

    def record_correct(self, qa: QAItem, propagate_info=None):
        qa_key = make_qa_key(qa)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = self._meta(qa)
        payload = {
            "label": LABEL_CORRECT,
            "human_answer": qa.answer,  # human agrees
            "option_index": None,
            "timestamp": now,
            "meta": meta,
        }
        if propagate_info:
            payload["auto_propagated"] = True
            payload["propagated_from"] = propagate_info

        self.latest[qa_key] = payload
        self._append_action({"type": "action", "timestamp": now, "qa_key": qa_key, **payload})
        self._write_checkpoint()
        self.dirty = True

    def record_corrected(self, qa: QAItem, option_index: int, propagate_info=None):
        qa_key = make_qa_key(qa)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = self._meta(qa)
        human_answer = qa.options[option_index] if 0 <= option_index < len(qa.options) else ""
        payload = {
            "label": LABEL_CORRECTED,
            "human_answer": human_answer,
            "option_index": int(option_index),
            "timestamp": now,
            "meta": meta,
        }
        if propagate_info:
            payload["auto_propagated"] = True
            payload["propagated_from"] = propagate_info

        self.latest[qa_key] = payload
        self._append_action({"type": "action", "timestamp": now, "qa_key": qa_key, **payload})
        self._write_checkpoint()
        self.dirty = True

    def _meta(self, qa: QAItem) -> Dict[str, Any]:
        return {
            "source_json": qa.source_json,
            "row_idx": qa.row_idx,
            "image_id": qa.image_id,
            "rgb_path": qa.rgb_path,
            "thermal_path": qa.thermal_path,
            "category": qa.category,
            "question_id": qa.question_id,
            "question": qa.question,
            "options": qa.options,
            "model_answer": qa.answer,
        }

    def write_report(self, total_rows: int, total_unique_images: int) -> str:
        evaluated = list(self.latest.values())
        n = len(evaluated)

        per_qid = defaultdict(lambda: Counter())
        per_cat = defaultdict(lambda: Counter())
        overall = Counter()
        corrections_map = defaultdict(Counter)

        for item in evaluated:
            label = item.get("label")
            meta = item.get("meta", {})
            qid = meta.get("question_id") or "(missing_question_id)"
            cat = meta.get("category") or "(missing_category)"
            model_ans = meta.get("model_answer", "")
            human_ans = item.get("human_answer", "")

            overall["evaluated"] += 1
            per_qid[qid]["evaluated"] += 1
            per_cat[cat]["evaluated"] += 1

            if label == LABEL_CORRECT:
                overall["correct"] += 1
                per_qid[qid]["correct"] += 1
                per_cat[cat]["correct"] += 1
            elif label == LABEL_CORRECTED:
                overall["wrong"] += 1
                per_qid[qid]["wrong"] += 1
                per_cat[cat]["wrong"] += 1
                corrections_map[qid][f"{model_ans}  ->  {human_ans}"] += 1

        def pct(a, b): return (100.0 * a / b) if b else 0.0

        qrows = []
        for qid, c in per_qid.items():
            ev = c["evaluated"]
            qrows.append((qid, ev, c["wrong"], pct(c["wrong"], ev), c["correct"], pct(c["correct"], ev)))
        qrows.sort(key=lambda x: (x[3], x[1]), reverse=True)

        lines = []
        lines.append("=" * 90)
        lines.append(f"MANUAL CORRECTION REPORT: {os.path.basename(self.report_txt)}")
        lines.append("=" * 90)
        lines.append(f"Actions JSONL:   {self.actions_jsonl}")
        lines.append(f"Checkpoint JSON: {self.checkpoint_json}")
        lines.append("")
        lines.append("DATASET:")
        lines.append(f"  Total QA rows loaded:        {total_rows}")
        lines.append(f"  Total unique images loaded:  {total_unique_images}")
        lines.append("")
        lines.append("EVALUATION:")
        lines.append(f"  Evaluated QA items: {n} ({pct(n, total_rows):.2f}% of rows)")
        lines.append(f"  Model==Human (Correct):      {overall['correct']} ({pct(overall['correct'], n):.2f}%)")
        lines.append(f"  Model!=Human (Incorrect):    {overall['wrong']} ({pct(overall['wrong'], n):.2f}%)")
        lines.append("")
        lines.append("-" * 90)
        lines.append("TOP QUESTIONS BY WRONG RATE (highest first):")
        lines.append("  question_id | evaluated | wrong | wrong% | correct | correct%")
        lines.append("  " + "-" * 84)
        for qid, ev, wrong, wrong_pct, cor, cor_pct in qrows[:30]:
            lines.append(f"  {str(qid)[:28]:28} | {ev:9d} | {wrong:5d} | {wrong_pct:6.2f}% | {cor:7d} | {cor_pct:7.2f}%")
        lines.append("")

        lines.append("-" * 90)
        lines.append("CATEGORY SUMMARY:")
        lines.append("  category | evaluated | wrong% | correct%")
        lines.append("  " + "-" * 84)
        for cat, c in sorted(per_cat.items(), key=lambda kv: kv[1]["evaluated"], reverse=True):
            ev = c["evaluated"]
            lines.append(f"  {str(cat)[:24]:24} | {ev:9d} | {pct(c['wrong'], ev):6.2f}% | {pct(c['correct'], ev):7.2f}%")
        lines.append("")

        lines.append("-" * 90)
        lines.append("COMMON CORRECTIONS (per question_id):")
        lines.append("  (shows the most frequent 'model_answer -> human_answer' mappings)")
        lines.append("  " + "-" * 84)
        for qid, _, _, _, _, _ in qrows[:20]:
            cm = corrections_map.get(qid)
            if not cm:
                continue
            lines.append(f"\n  QID: {qid}")
            for mapping, cnt in cm.most_common(5):
                lines.append(f"    {cnt:4d}x  {mapping}")
        lines.append("")
        lines.append("=" * 90)
        lines.append("END")
        lines.append("=" * 90)

        with open(self.report_txt, "w") as f:
            f.write("\n".join(lines))

        return self.report_txt


# -----------------------------
# UI
# -----------------------------
class ReviewerUI:
    def __init__(
        self,
        image_keys,
        groups,
        logger: EvalLogger,
        debug_keys=False,
        max_dim=900,
        cache_size=32,
        auto_advance=True,
        skip_done=False,
        max_option_buttons=6,
        qa_keys_in_sample: Optional[Set[str]] = None,
        group_map: Optional[Dict[str, str]] = None,
        all_items: Optional[List[QAItem]] = None,
    ):
        self.image_keys = image_keys
        self.groups = groups
        self.logger = logger
        self.debug_keys = debug_keys
        self.auto_advance = auto_advance
        self.skip_done = skip_done
        self.max_option_buttons = max_option_buttons
        self.qa_keys_in_sample: Set[str] = qa_keys_in_sample or set()
        
        # New: Auto-propagation support
        self.group_map = group_map or {}
        self.keyframe_to_items = defaultdict(list)
        
        if self.group_map and all_items:
            # Build reverse index: Keyframe -> [All QAItems in that group]
            print("[INFO] Building group index for auto-propagation...")
            for it in all_items:
                # Get the keyframe for this item's RGB path
                kf = self.group_map.get(it.rgb_path)
                if kf:
                    self.keyframe_to_items[kf].append(it)
            print(f"[INFO] Group index built. {len(self.keyframe_to_items)} keyframe groups.")

        self.img_i = 0
        self.qa_i = 0

        self.loader = ImageLoader(max_dim=max_dim, cache_size=cache_size)

        self.fig = plt.figure(figsize=(16, 8))
        self.fig.subplots_adjust(bottom=0.22)

        self.flash_text = self.fig.text(0.5, 0.97, "", ha="center", va="top", fontsize=12, fontweight="bold")
        self.flash_timer = None

        self.ax_rgb = self.fig.add_subplot(1, 3, 1)
        self.ax_thr = self.fig.add_subplot(1, 3, 2)
        self.ax_txt = self.fig.add_subplot(1, 3, 3)

        self.ax_rgb.set_title("RGB")
        self.ax_thr.set_title("Thermal")
        self.ax_txt.axis("off")

        self.rgb_artist = self.ax_rgb.imshow(Image.new("RGB", (10, 10)))
        self.thr_artist = self.ax_thr.imshow(Image.new("RGB", (10, 10)))
        self.rgb_msg = self.ax_rgb.text(0.5, 0.5, "", ha="center", va="center", wrap=True, transform=self.ax_rgb.transAxes)
        self.thr_msg = self.ax_thr.text(0.5, 0.5, "", ha="center", va="center", wrap=True, transform=self.ax_thr.transAxes)
        self.text_artist = self.ax_txt.text(0.0, 1.0, "", va="top", ha="left", wrap=True)

        self.ax_rgb.axis("off")
        self.ax_thr.axis("off")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._add_buttons()
        self._ensure_valid_start()
        self._update_save_button_state()
        self.render()

    def _add_buttons(self):
        def add_btn(axpos, label, cb):
            axb = self.fig.add_axes(axpos)
            b = Button(axb, label)
            b.on_clicked(lambda _evt: cb())
            return b

        self.btn_prevq = add_btn([0.02, 0.05, 0.09, 0.06], "Prev Q", self.prev_question)
        self.btn_nextq = add_btn([0.12, 0.05, 0.09, 0.06], "Next Q", self.next_question)
        self.btn_previ = add_btn([0.22, 0.05, 0.09, 0.06], "Prev Img", self.prev_image)
        self.btn_nexti = add_btn([0.32, 0.05, 0.09, 0.06], "Next Img", self.next_image)

        self.btn_correct = add_btn([0.44, 0.05, 0.12, 0.06], "✅ Correct (c)", self.mark_correct)

        # Option buttons (fixed grid 3x2 = 6)
        self.option_btns = []
        start_x = 0.58
        cols, rows = 3, 2
        w, h = 0.13, 0.06
        x_gap, y_gap = 0.01, 0.015
        y_bottom = 0.05
        y_top = y_bottom + h + y_gap

        positions = []
        for r in range(rows):
            y = y_top if r == 0 else y_bottom
            for c in range(cols):
                x = start_x + c * (w + x_gap)
                positions.append([x, y, w, h])

        max_buttons = min(self.max_option_buttons, len(positions))
        for i in range(max_buttons):
            axpos = positions[i]
            b = add_btn(axpos, f"{i+1}", lambda i=i: self.choose_option(i))
            self.option_btns.append(b)

        self.btn_save = add_btn([0.02, 0.13, 0.09, 0.05], "Save (s)", self.save_report)
        self.btn_quit = add_btn([0.12, 0.13, 0.09, 0.05], "Quit (q)", lambda: plt.close(self.fig))

    def _flash(self, msg: str, seconds: float = 1.2):
        try:
            if self.flash_timer is not None:
                self.flash_timer.stop()
        except Exception:
            pass

        self.flash_text.set_text(msg)
        self.flash_text.set_color("green")
        self.fig.canvas.draw_idle()

        def clear():
            self.flash_text.set_text("")
            self.fig.canvas.draw_idle()

        self.flash_timer = self.fig.canvas.new_timer(interval=int(seconds * 1000))
        self.flash_timer.add_callback(clear)
        self.flash_timer.start()

    def _update_save_button_state(self):
        if self.logger.dirty:
            self.btn_save.ax.set_facecolor((0.92, 0.92, 0.92, 1.0))
            self.btn_save.label.set_color("black")
        else:
            self.btn_save.ax.set_facecolor((0.85, 0.85, 0.85, 1.0))
            self.btn_save.label.set_color((0.4, 0.4, 0.4, 1.0))
        self.fig.canvas.draw_idle()

    def _reviewed_count(self) -> int:
        if not self.qa_keys_in_sample:
            return len(self.logger.latest)
        return sum(1 for k in self.logger.latest.keys() if k in self.qa_keys_in_sample)

    def _total_count(self) -> int:
        return len(self.qa_keys_in_sample) if self.qa_keys_in_sample else 0

    def _ensure_valid_start(self):
        if not self.skip_done:
            return
        self._advance_to_unevaluated_by_question(direction=+1, limit=len(self.image_keys) * 100)

    def current_list(self):
        return self.groups[self.image_keys[self.img_i]]

    def clamp_indices(self):
        self.img_i = max(0, min(self.img_i, len(self.image_keys) - 1))
        qlen = len(self.current_list())
        self.qa_i = max(0, min(self.qa_i, qlen - 1))

    def _find_question_in_image(self, img_index: int, target_question_id: str, fallback_index: int) -> int:
        key = self.image_keys[img_index]
        qa_list = self.groups[key]
        tq = (target_question_id or "").strip()
        if tq:
            for i, qa in enumerate(qa_list):
                if (qa.question_id or "").strip() == tq:
                    return i
        return max(0, min(fallback_index, len(qa_list) - 1))

    # -----------------------------
    # Skip logic
    # -----------------------------
    def _advance_to_unevaluated_by_question(self, direction: int, limit: int = 2000) -> bool:
        steps = 0
        while steps < limit:
            self.clamp_indices()
            qa = self.current_list()[self.qa_i]
            if not self.logger.is_done(qa):
                return True

            if direction > 0:
                if not self._next_question_internal():
                    return False
            else:
                if not self._prev_question_internal():
                    return False
            steps += 1
        return False

    def _advance_to_unevaluated_by_image(self, direction: int, limit: int = 2000) -> bool:
        steps = 0
        while steps < limit:
            self.clamp_indices()
            qa = self.current_list()[self.qa_i]
            if not self.logger.is_done(qa):
                return True

            if direction > 0:
                if not self._next_image_internal():
                    return False
            else:
                if not self._prev_image_internal():
                    return False
            steps += 1
        return False

    # -----------------------------
    # Auto Propagation Logic
    # -----------------------------
    def _propagate_label(self, source_qa: QAItem, label: str, option_index: Optional[int] = None):
        """
        Look up other images in the same group (same keyframe) with the same question_id
        and apply the same label.
        """
        if not self.group_map:
            return 0
        
        # 1. Get the keyframe for the current image
        keyframe = self.group_map.get(source_qa.rgb_path)
        if not keyframe:
            return 0
        
        # 2. Get all items in this group
        group_items = self.keyframe_to_items.get(keyframe, [])
        if not group_items:
            return 0
            
        count = 0
        prop_info = f"Source: {source_qa.rgb_path} (Keyframe: {keyframe})"
        
        for item in group_items:
            # Skip self
            if item.row_idx == source_qa.row_idx and item.source_json == source_qa.source_json:
                continue
                
            # Must match Question ID
            if item.question_id != source_qa.question_id:
                continue
                
            # Skip if already done
            if self.logger.is_done(item):
                continue
                
            # APPLY LABEL
            if label == LABEL_CORRECT:
                self.logger.record_correct(item, propagate_info=prop_info)
                count += 1
            elif label == LABEL_CORRECTED:
                # Ensure options are safe? 
                # Assuming if QuestionID matches, options match.
                self.logger.record_corrected(item, option_index, propagate_info=prop_info)
                count += 1
                
        return count

    # -----------------------------
    # Label actions
    # -----------------------------
    def mark_correct(self):
        self.clamp_indices()
        qa = self.current_list()[self.qa_i]
        self.logger.record_correct(qa)
        
        # Auto-propagate
        n = self._propagate_label(qa, LABEL_CORRECT)
        msg = "Correct!" if n == 0 else f"Correct! (Auto-labeled {n} duplicates)"
        self._flash(msg, seconds=1.5)

        self._update_save_button_state()

        if self.auto_advance:
            self.next_image()
        else:
            self.render()

    def choose_option(self, option_index: int):
        self.clamp_indices()
        qa = self.current_list()[self.qa_i]
        if not qa.options:
            return
        if option_index >= len(qa.options):
            return

        human_answer = qa.options[option_index]
        norm_human = norm_text(human_answer)
        norm_model = norm_text(qa.answer)

        label_to_apply = LABEL_CORRECT if (norm_human == norm_model) else LABEL_CORRECTED
        
        if label_to_apply == LABEL_CORRECT:
            self.logger.record_correct(qa)
        else:
            self.logger.record_corrected(qa, option_index)

        # Auto-propagate
        n = self._propagate_label(qa, label_to_apply, option_index)
        msg = "Recorded." if n == 0 else f"Recorded. (Auto-labeled {n} duplicates)"
        self._flash(msg, seconds=1.5)

        self._update_save_button_state()

        if self.auto_advance:
            self.next_image()
        else:
            self.render()

    # -----------------------------
    # Helper for "Global Jump"
    # -----------------------------
    def _jump_to_lowest_unevaluated(self, target_qa_index: int) -> bool:
        """
        Scans from Image 0 upwards.
        Finds the first image that has a Question at 'target_qa_index' which is NOT done.
        If found, moves the UI there and returns True.
        """
        for i, key in enumerate(self.image_keys):
            qa_list = self.groups[key]
            if target_qa_index < len(qa_list):
                qa = qa_list[target_qa_index]
                if not self.logger.is_done(qa):
                    self.img_i = i
                    self.qa_i = target_qa_index
                    return True
        return False

    def _jump_to_first_valid_image(self, target_qa_index: int):
        """
        Fallback: If ALL images are done for this question, 
        just go to the very first image that actually has this question,
        so the user can at least see it.
        """
        for i, key in enumerate(self.image_keys):
            qa_list = self.groups[key]
            if target_qa_index < len(qa_list):
                self.img_i = i
                self.qa_i = target_qa_index
                return

    # -----------------------------
    # Navigation (public)
    # -----------------------------
    def next_question(self):
        """
        Question-Centric Navigation:
        1. Calculate the NEW question index (current + 1).
        2. Scan the ENTIRE dataset from the beginning (Image 0).
        3. Jump to the first image where this new question is NOT evaluated.
        """
        target_qa_index = self.qa_i + 1
        
        # Try to find work to do for this new question
        found_work = self._jump_to_lowest_unevaluated(target_qa_index)
        
        if not found_work:
            # If no work is left for this question (or it doesn't exist yet),
            # we check if this question index is valid at all in the dataset.
            # We just jump to the first image that *has* this question so the user sees "All Done".
            self._jump_to_first_valid_image(target_qa_index)
            # If even that failed (index out of bounds), we stay put or stop.
        
        self.render()

    def prev_question(self):
        if self.qa_i > 0:
            target_qa_index = self.qa_i - 1
            found_work = self._jump_to_lowest_unevaluated(target_qa_index)
            if not found_work:
                self._jump_to_first_valid_image(target_qa_index)
        self.render()

    def next_image(self):
        if not self._next_image_internal():
            return
        if self.skip_done:
            # We jump over images that were just auto-labeled
            self._advance_to_unevaluated_by_image(direction=+1, limit=len(self.image_keys) * 200)
        self.render()

    def prev_image(self):
        if not self._prev_image_internal():
            return
        if self.skip_done:
            self._advance_to_unevaluated_by_image(direction=-1, limit=len(self.image_keys) * 200)
        self.render()

    # -----------------------------
    # Navigation internals
    # -----------------------------
    def _next_image_internal(self) -> bool:
        if self.img_i >= len(self.image_keys) - 1:
            return False
        current_qa = self.current_list()[self.qa_i]
        target_qid = (current_qa.question_id or "").strip()
        target_index = self.qa_i
        self.img_i += 1
        self.qa_i = self._find_question_in_image(self.img_i, target_qid, target_index)
        return True

    def _prev_image_internal(self) -> bool:
        if self.img_i <= 0:
            return False
        current_qa = self.current_list()[self.qa_i]
        target_qid = (current_qa.question_id or "").strip()
        target_index = self.qa_i
        self.img_i -= 1
        self.qa_i = self._find_question_in_image(self.img_i, target_qid, target_index)
        return True

    def _next_question_internal(self) -> bool:
        # Standard internal logic (used by skip logic, but not by main Next Question button anymore)
        qa_list = self.current_list()
        if self.qa_i < len(qa_list) - 1:
            self.qa_i += 1
            return True
        return False

    def _prev_question_internal(self) -> bool:
        if self.qa_i > 0:
            self.qa_i -= 1
            return True
        return False

    # -----------------------------
    # Save
    # -----------------------------
    def save_report(self):
        if not self.logger.dirty:
            self._flash("No new changes to save.", seconds=1.0)
            return

        total_rows = getattr(self, "_total_rows", 0)
        total_imgs = getattr(self, "_total_images", 0)
        path = self.logger.write_report(total_rows, total_imgs)
        print(f"[SAVED] {path}")

        self.logger.mark_clean()
        self._update_save_button_state()
        self._flash("Saved report ✅", seconds=1.2)

    # -----------------------------
    # Keys
    # -----------------------------
    def on_key(self, event):
        k = (event.key or "")
        if self.debug_keys:
            print(f"[KEY] raw='{k}'")
        k = k.lower()

        if k in ["n", "right", " ", "space"]:
            self.next_question()
        elif k in ["p", "left", "backspace"]:
            self.prev_question()
        elif k in ["j", "down"]:
            self.next_image()
        elif k in ["k", "up"]:
            self.prev_image()
        elif k == "c":
            self.mark_correct()
        elif k.isdigit():
            idx = int(k) - 1
            if idx >= 0:
                self.choose_option(idx)
        elif k == "s":
            self.save_report()
        elif k in ["q", "escape", "esc"]:
            plt.close(self.fig)

    # -----------------------------
    # Render
    # -----------------------------
    def render(self):
        try:
            self.clamp_indices()
            qa_list = self.current_list()
            qa = qa_list[self.qa_i]

            rgb = self.loader.get(qa.rgb_path)
            thr = self.loader.get(qa.thermal_path)

            if rgb is None:
                self.rgb_artist.set_data(Image.new("RGB", (10, 10)))
                self.rgb_msg.set_text(f"Missing/Unreadable\n{qa.rgb_path}")
            else:
                self.rgb_artist.set_data(rgb)
                self.rgb_msg.set_text("")

            if thr is None:
                self.thr_artist.set_data(Image.new("RGB", (10, 10)))
                self.thr_msg.set_text(f"Missing/Unreadable\n{qa.thermal_path}")
            else:
                self.thr_artist.set_data(thr)
                self.thr_msg.set_text("")

            opts = qa.options or []
            for i, b in enumerate(self.option_btns):
                if i < len(opts):
                    label = f"{i+1}: {opts[i]}"
                    if len(label) > 28:
                        label = label[:25] + "…"
                    b.label.set_text(label)
                    b.ax.set_visible(True)
                else:
                    b.label.set_text("")
                    b.ax.set_visible(False)

            done = self.logger.is_done(qa)
            status = ""
            if done:
                st = self.logger.latest.get(make_qa_key(qa), {})
                label_val = st.get("label")
                is_auto = st.get("auto_propagated", False)
                
                auto_str = " (AUTO-COPIED)" if is_auto else ""
                
                if label_val == LABEL_CORRECT:
                    status = f"STATUS: ✅ Correct{auto_str}"
                elif label_val == LABEL_CORRECTED:
                    status = f"STATUS: ❌ Corrected to: {st.get('human_answer','')}{auto_str}"
                else:
                    status = "STATUS: (evaluated)"

            header = f"Image {self.img_i+1}/{len(self.image_keys)} | Q {self.qa_i+1}/{len(qa_list)}"
            meta = (
                f"image_id: {qa.image_id}\ncategory: {qa.category}\nquestion_id: {qa.question_id}\n"
                f"source_json: {qa.source_json} | row_idx: {qa.row_idx}\n"
            )

            reviewed = self._reviewed_count()
            total = self._total_count()
            prog = ""
            if total > 0:
                prog = f"Reviewed: {reviewed} / {total} ({(100.0*reviewed/total):.2f}%)\n"

            opts_text = "\n".join([f"  {i+1}) {o}" for i, o in enumerate(opts)]) if opts else "  (no options)"
            body = (
                f"{header}\n"
                f"{prog}\n"
                f"{meta}"
                f"{status}\n\n"
                f"Question:\n{qa.question}\n\n"
                f"Options:\n{opts_text}\n\n"
                f"Model Answer:\n{qa.answer}\n\n"
                "NAV: n/right/space=next Q | p/left=prev Q | j/down=next img | k/up=prev img\n"
                "LABEL: c=Mark correct (AUTO-COPIES TO DUPLICATES)\n"
                "HUMAN ANSWER: press 1..6 or click (AUTO-COPIES TO DUPLICATES)\n"
                "SAVE: s=write report now | QUIT: q/Esc\n"
                f"(skip_done={self.skip_done}, auto_advance={self.auto_advance})\n"
            )
            self.text_artist.set_text(body)

            self.fig.canvas.draw_idle()
            plt.pause(0.001)

        except Exception as e:
            self.text_artist.set_text(f"Render error:\n{repr(e)}\n\nTry --max_dim 500 --cache_size 8.")
            self.fig.canvas.draw_idle()
            plt.pause(0.001)


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # --- Existing Arguments ---
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="Single JSON file")
    src.add_argument("--json_dir", help="Directory containing multiple JSON files")
    parser.add_argument("--group_map", type=str, help="JSON file from group_images.py")
    parser.add_argument("--sample_pct", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--debug_keys", action="store_true")
    parser.add_argument("--max_dim", type=int, default=600)
    parser.add_argument("--cache_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="manual_eval_outputs")
    parser.add_argument("--run_name", type=str, default="manual_eval")
    parser.add_argument("--eval_checkpoint", type=str, default=None, help="Existing checkpoint.json")
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--no_auto_advance", action="store_true")
    parser.add_argument("--max_option_buttons", type=int, default=6)
    
    # --- NEW ARGUMENT ---
    parser.add_argument("--remaining", type=str, help="JSON file containing list of specific image paths to evaluate.")

    args = parser.parse_args()

    # 1. Load All Items
    if args.json_dir:
        items = load_items_from_dir(args.json_dir)
        source_desc = f"dir:{args.json_dir}"
    else:
        items = load_items(args.json)
        source_desc = f"file:{args.json}"

    if not items:
        raise SystemExit("No QA items found.")

    # 2. Validate Checkpoint (CRITICAL STEP)
    # We identify which Question IDs exist in the checkpoint to ensure we don't mix them up.
    valid_qids = set()
    if args.eval_checkpoint and os.path.exists(args.eval_checkpoint):
        try:
            with open(args.eval_checkpoint, "r") as f:
                ckpt_data = json.load(f)
                ckpt_items = ckpt_data.get("items", {})
                for k, v in ckpt_items.items():
                    # Extract question_id from the saved metadata
                    meta = v.get("meta", {})
                    if "question_id" in meta:
                        valid_qids.add(str(meta["question_id"]).strip())
            
            if valid_qids:
                print(f"[INFO] Restricted to {len(valid_qids)} Question IDs found in checkpoint.")
        except Exception as e:
            print(f"[WARN] Could not read checkpoint for validation: {e}")

    # 3. Load Remaining Images Filter
    remaining_paths = None
    if args.remaining:
        if os.path.exists(args.remaining):
            with open(args.remaining, "r") as f:
                # Load set of paths, stripping whitespace to be safe
                remaining_paths = set(p.strip() for p in json.load(f))
            print(f"[INFO] Filtering for {len(remaining_paths)} images from {args.remaining}")
        else:
            raise FileNotFoundError(f"Remaining images file not found: {args.remaining}")

    # 4. Apply Filters
    # This ensures the human ONLY sees the correct images and Correct Question IDs
    filtered_items = []
    for it in items:
        # Filter 1: Must be in remaining_images.json (if flag provided)
        if remaining_paths is not None:
            if it.rgb_path.strip() not in remaining_paths:
                continue
        
        # Filter 2: Must match Question IDs in checkpoint (if checkpoint has data)
        # This prevents the user from adding new Question IDs to an existing specialized checkpoint.
        if valid_qids:
            if str(it.question_id).strip() not in valid_qids:
                continue
        
        filtered_items.append(it)

    if not filtered_items:
        print("[ERROR] No items match the combined criteria (Remaining list + Checkpoint Question IDs).")
        return

    # --- Proceed with existing logic using filtered_items ---
    
    # Load Group Map
    group_map = {}
    if args.group_map:
        if os.path.exists(args.group_map):
            with open(args.group_map, "r") as f:
                group_map = json.load(f)
            print(f"Loaded grouping map with {len(group_map)} entries.")
        else:
            print(f"[WARN] Group map file not found: {args.group_map}")

    order, groups = group_by_image(filtered_items)

    pct = max(0.0, min(100.0, normalize_sample_pct(args.sample_pct)))
    n_total_images = len(order)
    
    # Since we are in "Remaining" mode, we likely want to see ALL of them, 
    # but we respect the sample_pct if the user provided it.
    n_pick = max(1, int(math.ceil((pct / 100.0) * n_total_images))) if pct > 0 else 0
    
    rng = random.Random(args.seed)
    sampled = rng.sample(order, k=min(n_pick, n_total_images))
    sampled_set = set(sampled)
    sampled_ordered = [k for k in order if k in sampled_set]

    qa_keys_in_sample: Set[str] = set()
    for img_key in sampled_ordered:
        for qa in groups[img_key]:
            qa_keys_in_sample.add(make_qa_key(qa))

    print(f"Source: {source_desc}")
    print(f"Loaded {len(filtered_items)} QA rows across {n_total_images} unique images.")
    
    logger = EvalLogger(out_dir=args.out_dir, run_name=args.run_name, checkpoint_in=args.eval_checkpoint)
    
    ui = ReviewerUI(
        sampled_ordered,
        groups,
        logger=logger,
        debug_keys=args.debug_keys,
        max_dim=args.max_dim,
        cache_size=args.cache_size,
        auto_advance=(not args.no_auto_advance),
        skip_done=args.skip_done,
        max_option_buttons=args.max_option_buttons,
        qa_keys_in_sample=qa_keys_in_sample,
        group_map=group_map,
        all_items=filtered_items 
    )
    ui._total_rows = len(filtered_items)
    ui._total_images = n_total_images

    plt.show()

    report_path = logger.write_report(total_rows=len(filtered_items), total_unique_images=n_total_images)
    print(f"[FINAL] Report: {report_path}")

if __name__ == "__main__":
    main()