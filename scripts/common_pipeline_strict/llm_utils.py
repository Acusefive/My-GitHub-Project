from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib import error, request

from .io_utils import ensure_dir


SummaryKey = Tuple[str, int, str]


def summary_cache_key(user_id: str, target_t: int, target_pid: str) -> str:
    return f"{user_id}\t{int(target_t)}\t{target_pid}"


def load_summary_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    cache: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(item.get("key") or "").strip()
            value = str(item.get("summary_text") or "").strip()
            if key:
                cache[key] = value
    return cache


def append_summary_cache(path: Path, key: str, summary_text: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "summary_text": summary_text}, ensure_ascii=False) + "\n")


class OpenAICompatibleSummarizer:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str],
        timeout_sec: int,
        max_tokens: int,
        temperature: float,
        retries: int = 3,
    ) -> None:
        if not base_url:
            raise ValueError("LLM summary enabled but llm_base_url is empty")
        if not model:
            raise ValueError("LLM summary enabled but llm_model is empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = (api_key or "").strip()
        self.timeout_sec = int(timeout_sec)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.retries = int(retries)

    def summarize(
        self,
        *,
        target_pid: str,
        target_question_text: str,
        target_semantic_id: str,
        target_concepts: Iterable[str],
        evidence_list: Iterable[Dict[str, object]],
        template_summary_text: str,
    ) -> str:
        evidence_lines = []
        for idx, evidence in enumerate(evidence_list, start=1):
            evidence_lines.append(
                (
                    f"{idx}. role={evidence.get('role', '')}; "
                    f"overlap={evidence.get('knowledge_overlap', '')}; "
                    f"level_diff={evidence.get('level_diff', '')}; "
                    f"answer={evidence.get('answer_result', '')}; "
                    f"score={evidence.get('support_score', '')}; "
                    f"text={evidence.get('question_text', '')}"
                )
            )

        prompt = (
            "你是一个教育认知状态压缩器。\n\n"
            "任务：\n"
            "根据输入的目标题目、学生近期作答统计、以及已筛选的认知证据，生成“结构化认知摘要对象”，用于描述学生当前的认知状态。\n\n"
            "要求：\n"
            "1. 只描述当前认知状态，不给建议，不评价教学，不鼓励，不解释推理过程，不输出<think>标签。\n"
            "2. 摘要必须围绕以下维度：\n"
            "   - 稳定掌握点\n"
            "   - 持续薄弱点\n"
            "   - 近期波动性\n"
            "3. 表述必须贴近认知状态，不要写成教师评语。禁止使用“建议、需要、应当、继续、加强、多做训练、应加强”等措辞。\n"
            "4. 只能依据输入证据生成内容；若证据不足，可保守表述，但不得编造。\n"
            "5. 输出必须是一个JSON对象，且只能包含以下字段：\n"
            '   - stable_points: 字符串数组，0到2项\n'
            '   - weak_points: 字符串数组，0到2项\n'
            '   - volatility: 只能取“低”“中”“高”\n'
            '   - confidence: 只能取“低”“中”“高”\n'
            '   - summary: 1到2句中文，总长度不超过60字\n'
            "6. summary 应优先概括：\n"
            "   - 哪些基础能力较稳定\n"
            "   - 哪些关键薄弱点持续存在\n"
            "   - 近期是否有明显波动\n"
            "7. 不要复述输入原文，不要逐条列证据，不要输出额外字段，不要输出Markdown代码块。\n\n"
            f"目标题ID: {target_pid}\n"
            f"目标题语义ID: {target_semantic_id}\n"
            f"目标题文本: {target_question_text}\n"
            f"目标题知识点: {'、'.join(str(x) for x in target_concepts)}\n"
            f"学生近期模板统计摘要: {template_summary_text}\n"
            "认知证据:\n"
            + "\n".join(evidence_lines)
        )

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你只输出一个合法JSON对象。"
                        "不得输出<think>、解释、分析过程、Markdown代码块或额外文本。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            req = request.Request(
                url=self.base_url + "/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
                text = self._postprocess_content(self._flatten_content(content))
                if text:
                    return text
                raise ValueError("Empty summary returned from LLM")
            except (error.URLError, error.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(5, attempt))
        raise RuntimeError(f"Failed to get LLM summary after {self.retries} attempts: {last_error}")

    @staticmethod
    def _flatten_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text") or ""))
                else:
                    chunks.append(str(item))
            return "".join(chunks)
        return str(content)

    @staticmethod
    def _postprocess_content(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"^<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()
        cleaned = cleaned.strip("` \n\r\t")
        if not cleaned:
            return ""

        json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if json_match:
            candidate = json_match.group(0).strip()
            obj = json.loads(candidate)
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = cleaned[first : last + 1].strip()
            obj = json.loads(candidate)
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

        obj = json.loads(cleaned)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
