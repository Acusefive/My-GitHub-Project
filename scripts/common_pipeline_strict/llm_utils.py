from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error, request

from .io_utils import ensure_dir


SummaryKey = Tuple[str, int, str]
_LEVEL_VALUES = {"低", "中", "高"}


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


def append_summary_cache_entries(path: Path, entries: Iterable[Tuple[str, str]]) -> None:
    rows = [(str(key), str(summary_text)) for key, summary_text in entries if str(key).strip()]
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for key, summary_text in rows:
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
    expected_keys = {
        "mastered_concepts",
        "weak_concepts",
        "transfer_state",
        "risk_level",
        "evidence_quality",
        "diagnosis",
    }
    if set(parsed.keys()) != expected_keys:
        raise ValueError(f"Unexpected llm summary fields: {sorted(parsed.keys())}")

    def _concept_items(name: str) -> List[Dict[str, Any]]:
        raw_items = parsed.get(name)
        if not isinstance(raw_items, list):
            raise ValueError(f"{name} must be an array")
        items: List[Dict[str, Any]] = []
        for item in raw_items[:3]:
            if not isinstance(item, dict):
                raise ValueError(f"{name} items must be objects")
            concept = str(item.get("concept") or "").strip()
            evidence_ids_raw = item.get("evidence_ids") or []
            confidence = str(item.get("confidence") or "").strip()
            if not concept:
                continue
            if not isinstance(evidence_ids_raw, list):
                raise ValueError(f"{name}.evidence_ids must be an array")
            evidence_ids: List[int] = []
            for evidence_id in evidence_ids_raw[:6]:
                try:
                    evidence_ids.append(int(evidence_id))
                except Exception:
                    continue
            if confidence not in _LEVEL_VALUES:
                raise ValueError(f"Invalid {name}.confidence: {confidence}")
            items.append(
                {
                    "concept": concept,
                    "evidence_ids": evidence_ids,
                    "confidence": confidence,
                }
            )
        return items

    mastered_concepts = _concept_items("mastered_concepts")
    weak_concepts = _concept_items("weak_concepts")
    transfer_state = str(parsed.get("transfer_state") or "").strip()
    risk_level = str(parsed.get("risk_level") or "").strip()
    evidence_quality = str(parsed.get("evidence_quality") or "").strip()
    diagnosis = str(parsed.get("diagnosis") or "").strip()
    if risk_level not in _LEVEL_VALUES:
        raise ValueError(f"Invalid risk_level: {risk_level}")
    if evidence_quality not in _LEVEL_VALUES:
        raise ValueError(f"Invalid evidence_quality: {evidence_quality}")
    if not transfer_state:
        raise ValueError("transfer_state is empty")
    if not diagnosis:
        raise ValueError("diagnosis is empty")

    return {
        "mastered_concepts": mastered_concepts,
        "weak_concepts": weak_concepts,
        "transfer_state": transfer_state,
        "risk_level": risk_level,
        "evidence_quality": evidence_quality,
        "diagnosis": diagnosis,
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
        evidence_lines: List[str] = []
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
            "任务：根据目标题目、学生近期作答统计、以及已筛选的认知证据，生成结构化认知摘要对象。\n\n"
            "严格要求：\n"
            "1. 只能输出一个 JSON 对象。\n"
            "2. JSON 必须且只能包含 stable_points, weak_points, volatility, confidence, summary 这 5 个字段。\n"
            "3. stable_points 和 weak_points 都必须出现；如果没有内容，必须输出空数组 []。\n"
            "4. stable_points 和 weak_points 必须是字符串数组，长度只能是 0、1、2。\n"
            "5. 如果候选稳定点或薄弱点超过 2 个，只保留最关键的 2 个。\n"
            "6. volatility 和 confidence 只能取“低”“中”“高”。\n"
            "7. summary 必须是 1 到 2 句中文，总长度不超过 60 字，且不能为空。\n"
            "8. 即使 stable_points 和 weak_points 都为空，summary 也必须给出一句保守中文概括。\n"
            "9. 只描述当前认知状态，不给建议，不评价教学，不鼓励，不解释推理过程。\n"
            "10. 不要输出 <think>、Markdown 代码块或任何额外文本。\n"
            "11. 不能编造输入中不存在的知识点或证据结论。\n\n"
            "合法输出示例 1：\n"
            '{"stable_points":["基础概念辨识"],"weak_points":["层级迁移"],"volatility":"中","confidence":"中","summary":"基础概念辨识较稳定，层级迁移仍偏薄弱，近期表现存在波动。"}\n'
            "合法输出示例 2：\n"
            '{"stable_points":[],"weak_points":[],"volatility":"中","confidence":"低","summary":"当前证据有限，稳定掌握点与持续薄弱点暂不明显，近期波动中等。"}\n\n'
            f"目标题ID: {target_pid}\n"
            f"目标题语义ID: {target_semantic_id}\n"
            f"目标题文本: {target_question_text}\n"
            f"目标题知识点: {'、'.join(str(x) for x in target_concepts)}\n"
            f"学生近期模板统计摘要: {template_summary_text}\n\n"
            "认知证据:\n"
            + "\n".join(evidence_lines)
        )

        system_prompt = (
            "你只输出一个合法 JSON 对象。"
            "必须同时包含 stable_points、weak_points、volatility、confidence、summary 这 5 个字段。"
            "stable_points 和 weak_points 即使没有内容也必须输出为空数组 []。"
            "summary 绝不能是空字符串。"
            "不得输出 <think>、解释、分析过程、Markdown 代码块或额外文本。"
        )

        current_user_prompt = prompt
        last_exc: Optional[Exception] = None
        last_raw_json = ""
        for attempt in range(3):
            obj = self.request_json(system_prompt=system_prompt, user_prompt=current_user_prompt)
            raw_json = json.dumps(obj, ensure_ascii=False)
            last_raw_json = raw_json
            try:
                parsed = parse_llm_summary_json(raw_json)
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    break
                current_user_prompt = (
                    "你上一次输出的 JSON 不合法，请严格修复。\n"
                    "修复要求：\n"
                    "1. 必须同时包含 stable_points, weak_points, volatility, confidence, summary 这 5 个字段。\n"
                    "2. stable_points 和 weak_points 必须是数组，即使没有内容也必须输出 []。\n"
                    "3. stable_points 和 weak_points 的长度都不能超过 2。\n"
                    "4. summary 不能为空；即使稳定点和薄弱点都为空，也必须输出一句保守中文摘要。\n"
                    "5. 不能新增额外字段，也不能缺少字段。\n"
                    "6. volatility 和 confidence 只能取“低”“中”“高”。\n"
                    "7. 只输出修复后的 JSON 对象，不要输出解释。\n"
                    "8. 合法格式示例：{\"stable_points\":[],\"weak_points\":[],\"volatility\":\"中\",\"confidence\":\"低\",\"summary\":\"当前证据有限，稳定掌握点与持续薄弱点暂不明显，近期波动中等。\"}\n\n"
                    f"原始非法 JSON:\n{raw_json}\n\n"
                    f"错误原因: {exc}"
                )

        hard_schema_prompt = (
            "请直接填写下面这个 JSON 模板，不能删除字段，不能增加字段，不能返回空对象：\n"
            "{\"stable_points\":[],\"weak_points\":[],\"volatility\":\"中\",\"confidence\":\"低\",\"summary\":\"当前证据有限，稳定掌握点与持续薄弱点暂不明显，近期波动中等。\"}\n\n"
            "填写规则：\n"
            "1. stable_points 和 weak_points 必须保留为数组，最多各 2 项，没有内容就保持 []。\n"
            "2. volatility 和 confidence 只能取“低”“中”“高”。\n"
            "3. summary 必须是 1 到 2 句中文，不超过 60 字，且不能为空。\n"
            "4. 如果 evidence 很弱或目标语义异常，也必须输出一句保守中文摘要，不能留空。\n"
            "5. 只能依据输入证据，不得编造。\n"
            "6. 只输出填写后的 JSON 对象。\n\n"
            f"目标题ID: {target_pid}\n"
            f"目标题语义ID: {target_semantic_id}\n"
            f"目标题文本: {target_question_text}\n"
            f"目标题知识点: {'、'.join(str(x) for x in target_concepts)}\n"
            f"学生近期模板统计摘要: {template_summary_text}\n"
            "认知证据:\n"
            + "\n".join(evidence_lines)
        )
        try:
            obj = self.request_json(system_prompt=system_prompt, user_prompt=hard_schema_prompt)
            raw_json = json.dumps(obj, ensure_ascii=False)
            last_raw_json = raw_json
            parsed = parse_llm_summary_json(raw_json)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            last_exc = exc

        raise ValueError(
            f"Invalid LLM summary for target_pid={target_pid}, target_semantic_id={target_semantic_id}: {last_exc}. "
            f"raw_obj={last_raw_json}"
        ) from last_exc


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
        evidence_lines: List[str] = []
        for idx, evidence in enumerate(evidence_list, start=1):
            raw_scores = evidence.get("raw_scores", {})
            activation = evidence.get("activation", {})
            evidence_lines.append(
                (
                    f"{idx}. evidence_id={idx}; "
                    f"hist_pid={evidence.get('problem_id', '')}; "
                    f"hist_semantic_id={evidence.get('semantic_id', '')}; "
                    f"history_pos={evidence.get('history_pos', '')}; "
                    f"role={evidence.get('role', '')}; "
                    f"overlap={evidence.get('knowledge_overlap', '')}; "
                    f"level_diff={evidence.get('level_diff', '')}; "
                    f"answer={evidence.get('answer_result', '')}; "
                    f"support_score={evidence.get('support_score', '')}; "
                    f"activation={json.dumps(activation, ensure_ascii=False)}; "
                    f"raw_scores={json.dumps(raw_scores, ensure_ascii=False)}; "
                    f"text={evidence.get('question_text', '')}"
                )
            )

        prompt = (
            "你是一个教育认知诊断压缩器。\n\n"
            "任务：根据目标题、学生近期模板统计摘要、以及已筛选的认知证据，生成结构化诊断 JSON。\n\n"
            "严格要求：\n"
            "1. 只能输出一个 JSON 对象，不要输出 Markdown、解释或 <think>。\n"
            "2. JSON 必须且只能包含 mastered_concepts, weak_concepts, transfer_state, risk_level, evidence_quality, diagnosis 这 6 个字段。\n"
            "3. mastered_concepts 和 weak_concepts 必须是数组，最多各 3 项；没有内容时输出 []。\n"
            "4. mastered_concepts/weak_concepts 的每一项必须是对象：{\"concept\":字符串,\"evidence_ids\":整数数组,\"confidence\":\"低/中/高\"}。\n"
            "5. concept 只能来自目标题知识点或证据 overlap，不得编造输入中不存在的知识点。\n"
            "6. evidence_ids 必须引用输入证据里的 evidence_id；若证据不足可为空数组。\n"
            "7. risk_level 和 evidence_quality 只能取 低、中、高。\n"
            "8. transfer_state 用一个短语概括迁移状态，例如：同质迁移稳定、前置不足、高阶迁移风险、协同证据不足、证据有限。\n"
            "9. diagnosis 必须是 1 到 2 句中文，总长度不超过 80 字，且不能为空。\n"
            "10. 如果证据冲突、主要依赖协同、或知识重合弱，应降低 evidence_quality 或 confidence。\n\n"
            "合法输出示例：\n"
            '{"mastered_concepts":[{"concept":"放大电路","evidence_ids":[1,3],"confidence":"高"}],"weak_concepts":[{"concept":"输出电阻","evidence_ids":[2],"confidence":"中"}],"transfer_state":"同质迁移稳定","risk_level":"中","evidence_quality":"高","diagnosis":"放大电路相关证据较稳定，但输出电阻迁移仍有风险。"}\n\n'
            f"目标题ID: {target_pid}\n"
            f"目标题语义ID: {target_semantic_id}\n"
            f"目标题文本: {target_question_text}\n"
            f"目标题知识点: {'、'.join(str(x) for x in target_concepts)}\n"
            f"学生近期模板统计摘要: {template_summary_text}\n\n"
            "认知证据:\n"
            + "\n".join(evidence_lines)
        )
        system_prompt = (
            "你只输出一个合法 JSON 对象。"
            "必须同时包含 mastered_concepts、weak_concepts、transfer_state、risk_level、evidence_quality、diagnosis 这 6 个字段。"
            "不得输出 <think>、解释、分析过程、Markdown 代码块或额外文本。"
        )

        current_user_prompt = prompt
        last_exc: Optional[Exception] = None
        last_raw_json = ""
        for attempt in range(3):
            obj = self.request_json(system_prompt=system_prompt, user_prompt=current_user_prompt)
            raw_json = json.dumps(obj, ensure_ascii=False)
            last_raw_json = raw_json
            try:
                parsed = parse_llm_summary_json(raw_json)
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    break
                current_user_prompt = (
                    "你上一次输出的 JSON 不合法，请严格修复。\n"
                    "必须只包含 mastered_concepts, weak_concepts, transfer_state, risk_level, evidence_quality, diagnosis 这 6 个字段；"
                    "mastered_concepts 和 weak_concepts 必须是数组；risk_level/evidence_quality 只能是 低、中、高；diagnosis 不能为空。\n\n"
                    f"原始非法 JSON:\n{raw_json}\n\n"
                    f"错误原因: {exc}"
                )

        hard_schema_prompt = (
            "请直接填写下面这个 JSON 模板，不得增加或删除字段：\n"
            "{\"mastered_concepts\":[],\"weak_concepts\":[],\"transfer_state\":\"证据有限\",\"risk_level\":\"中\",\"evidence_quality\":\"低\",\"diagnosis\":\"当前证据有限，稳定掌握点与薄弱点暂不明显。\"}\n\n"
            "只能依据输入证据填写；concept 只能来自目标题知识点或 overlap；evidence_ids 必须引用证据编号。\n\n"
            f"目标题ID: {target_pid}\n"
            f"目标题语义ID: {target_semantic_id}\n"
            f"目标题文本: {target_question_text}\n"
            f"目标题知识点: {'、'.join(str(x) for x in target_concepts)}\n"
            f"学生近期模板统计摘要: {template_summary_text}\n"
            "认知证据:\n"
            + "\n".join(evidence_lines)
        )
        try:
            obj = self.request_json(system_prompt=system_prompt, user_prompt=hard_schema_prompt)
            raw_json = json.dumps(obj, ensure_ascii=False)
            last_raw_json = raw_json
            parsed = parse_llm_summary_json(raw_json)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            last_exc = exc

        raise ValueError(
            f"Invalid LLM diagnosis for target_pid={target_pid}, target_semantic_id={target_semantic_id}: {last_exc}. "
            f"raw_obj={last_raw_json}"
        ) from last_exc


class OpenAICompatibleGraphCompleter(OpenAICompatibleJsonClient):
    def complete(
        self,
        *,
        concept: str,
        chapters: Iterable[str],
        candidate_concepts: Iterable[str],
    ) -> Dict[str, Any]:
        candidates = [
            str(item).strip()
            for item in candidate_concepts
            if str(item).strip() and str(item).strip() != concept
        ]
        prompt = (
            "你是一个教育知识图谱构建器。\n"
            "任务：根据一个知识点和候选知识点列表，判断其中哪些是该知识点的前置支撑点，哪些是强相关的邻近/同域知识点。\n"
            "要求：\n"
            "1. 只能从候选列表中选择，不得编造新知识点。\n"
            "2. prerequisite_candidates 表示前置支撑知识点，0 到 2 项。\n"
            "3. related_candidates 表示强相关知识点，0 到 3 项。\n"
            "4. confidence 只能是“低”“中”“高”。\n"
            "5. 只能输出 JSON 对象，包含字段：prerequisite_candidates, related_candidates, confidence。\n\n"
            f"目标知识点: {concept}\n"
            f"所在章节: {'、'.join(str(item) for item in chapters if str(item).strip()) or '无可用章节信息'}\n"
            f"候选知识点: {'、'.join(candidates)}\n"
        )
        obj = self.request_json(
            system_prompt=(
                "你只输出一个合法 JSON 对象。"
                "不得输出 <think>、解释、分析过程、Markdown 代码块或额外文本。"
            ),
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
        if confidence not in _LEVEL_VALUES:
            raise ValueError(f"Invalid graph completion confidence: {confidence}")
        return {
            "prerequisite_candidates": prereq,
            "related_candidates": related,
            "confidence": confidence,
        }
