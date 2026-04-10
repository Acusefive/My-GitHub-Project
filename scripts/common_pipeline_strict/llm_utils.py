from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
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


def load_json_cache(path: Path, *, value_field: str = "payload") -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(item.get("key") or "").strip()
            payload = item.get(value_field)
            if key and isinstance(payload, dict):
                cache[key] = payload
    return cache


def append_json_cache(path: Path, key: str, payload: Dict[str, Any], *, value_field: str = "payload") -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, value_field: payload}, ensure_ascii=False) + "\n")


def parse_llm_summary_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("llm_summary_text is empty")
    parsed = json.loads(raw)
    expected_keys = {"stable_points", "weak_points", "volatility", "confidence", "summary"}
    if set(parsed.keys()) != expected_keys:
        raise ValueError(f"Unexpected llm summary fields: {sorted(parsed.keys())}")
    stable_raw = parsed.get("stable_points")
    weak_raw = parsed.get("weak_points")
    if not isinstance(stable_raw, list) or not isinstance(weak_raw, list):
        raise ValueError("stable_points/weak_points must be arrays")
    stable_points = [str(item).strip() for item in stable_raw if str(item).strip()]
    weak_points = [str(item).strip() for item in weak_raw if str(item).strip()]
    if len(stable_points) > 2 or len(weak_points) > 2:
        raise ValueError("stable_points/weak_points exceed max length 2")
    volatility = str(parsed.get("volatility")).strip()
    confidence = str(parsed.get("confidence")).strip()
    summary = str(parsed.get("summary")).strip()
    if volatility not in {"低", "中", "高"}:
        raise ValueError(f"Invalid volatility: {volatility}")
    if confidence not in {"低", "中", "高"}:
        raise ValueError(f"Invalid confidence: {confidence}")
    if not summary:
        raise ValueError("summary is empty")
    return {
        "stable_points": stable_points,
        "weak_points": weak_points,
        "volatility": volatility,
        "confidence": confidence,
        "summary": summary,
    }


class OpenAICompatibleJsonClient:
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
            raise ValueError("LLM client base_url is empty")
        if not model:
            raise ValueError("LLM client model is empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = (api_key or "").strip()
        self.timeout_sec = int(timeout_sec)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.retries = int(retries)

    def request_json(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
                if not text:
                    raise ValueError("Empty JSON response returned from LLM")
                return json.loads(text)
            except (error.URLError, error.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(5, attempt))
        raise RuntimeError(f"Failed to get LLM JSON after {self.retries} attempts: {last_error}")

    @staticmethod
    def _flatten_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
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


class OpenAICompatibleSummarizer(OpenAICompatibleJsonClient):
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
            "   - stable_points: 字符串数组，0到2项\n"
            "   - weak_points: 字符串数组，0到2项\n"
            "   - volatility: 只能取“低”“中”“高”\n"
            "   - confidence: 只能取“低”“中”“高”\n"
            "   - summary: 1到2句中文，总长度不超过60字\n"
            "6. summary 应优先概括：\n"
            "   - 哪些基础能力较稳定\n"
            "   - 哪些关键薄弱点持续存在\n"
            "   - 近期是否有明显波动\n"
            "7. 不要复述输入原文，不要逐条列证据，不要输出额外字段，不要输出Markdown代码块。\n\n"
            f"目标题ID: {target_pid}\n"
            f"目标题语义ID: {target_semantic_id}\n"
            f"目标题文本: {target_question_text}\n"
            f"目标题知识点: {'、'.join(str(x) for x in target_concepts)}\n"
            f"学生近期模板统计摘要: {template_summary_text}\n\n"
            "认知证据:\n"
            + "\n".join(evidence_lines)
        )
        obj = self.request_json(
            system_prompt="你只输出一个合法JSON对象。不得输出<think>、解释、分析过程、Markdown代码块或额外文本。",
            user_prompt=prompt,
        )
        return json.dumps(parse_llm_summary_json(json.dumps(obj, ensure_ascii=False)), ensure_ascii=False, separators=(",", ":"))


class OpenAICompatibleGraphCompleter(OpenAICompatibleJsonClient):
    def complete(
        self,
        *,
        concept: str,
        chapters: Iterable[str],
        candidate_concepts: Iterable[str],
    ) -> Dict[str, Any]:
        candidates = [str(item).strip() for item in candidate_concepts if str(item).strip() and str(item).strip() != concept]
        prompt = (
            "你是一个教育知识图谱构建器。\n"
            "任务：根据一个知识点和候选知识点列表，判断其中哪些是该知识点的前置支撑点，哪些是强相关的同域/邻近知识点。\n"
            "要求：\n"
            "1. 只能从候选列表中选择，不得编造新知识点。\n"
            "2. prerequisite_candidates 表示对当前知识点有前置支撑意义的知识点，0到2项。\n"
            "3. related_candidates 表示强相关的邻近/同域知识点，0到3项。\n"
            "4. confidence 只能是“低”“中”“高”。\n"
            "5. 只能输出 JSON 对象，包含字段：prerequisite_candidates, related_candidates, confidence。\n\n"
            f"目标知识点: {concept}\n"
            f"所在章节: {'、'.join(str(item) for item in chapters if str(item).strip()) or '无可用章节信息'}\n"
            f"候选知识点: {'、'.join(candidates)}\n"
        )
        obj = self.request_json(
            system_prompt="你只输出一个合法JSON对象。不得输出<think>、解释、分析过程、Markdown代码块或额外文本。",
            user_prompt=prompt,
        )
        expected_keys = {"prerequisite_candidates", "related_candidates", "confidence"}
        if set(obj.keys()) != expected_keys:
            raise ValueError(f"Unexpected llm graph completion fields: {sorted(obj.keys())}")
        prereq_raw = obj.get("prerequisite_candidates")
        related_raw = obj.get("related_candidates")
        if not isinstance(prereq_raw, list) or not isinstance(related_raw, list):
            raise ValueError("Graph completion candidate fields must be arrays")
        prereq = [str(item).strip() for item in prereq_raw if str(item).strip() and str(item).strip() != concept][:2]
        related = [str(item).strip() for item in related_raw if str(item).strip() and str(item).strip() != concept][:3]
        confidence = str(obj.get("confidence")).strip()
        if confidence not in {"低", "中", "高"}:
            raise ValueError(f"Invalid graph completion confidence: {confidence}")
        return {
            "prerequisite_candidates": prereq,
            "related_candidates": related,
            "confidence": confidence,
        }
