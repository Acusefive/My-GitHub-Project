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
        disable_thinking: bool = False,
        use_chat_template_kwargs: bool = False,
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
        self.disable_thinking = bool(disable_thinking)
        self._allow_chat_template_kwargs = bool(disable_thinking and use_chat_template_kwargs)

    def request_json(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.request_json_with_raw(system_prompt=system_prompt, user_prompt=user_prompt)["parsed"]

    def request_json_with_raw(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            body = self._request_body(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                include_chat_template_kwargs=self._allow_chat_template_kwargs,
            )
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
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
                content_text = self._flatten_content(content)
                text = self._postprocess_content(content_text)
                if not text:
                    raise ValueError("Empty JSON response returned from LLM")
                return {
                    "parsed": json.loads(text),
                    "raw_response": raw,
                    "content": content_text,
                    "json_text": text,
                    "attempts": attempt,
                }
            except (error.URLError, error.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                if (
                    isinstance(exc, error.HTTPError)
                    and self._allow_chat_template_kwargs
                    and int(getattr(exc, "code", 0) or 0) in {400, 422}
                ):
                    self._allow_chat_template_kwargs = False
                if attempt >= self.retries:
                    break
                time.sleep(min(5, attempt))
        raise RuntimeError(f"Failed to get LLM JSON after {self.retries} attempts: {last_error}")

    def _request_body(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        include_chat_template_kwargs: bool,
    ) -> Dict[str, Any]:
        user_content = str(user_prompt)
        if self.disable_thinking and "/no_think" not in user_content:
            user_content = user_content.rstrip() + "\n/no_think"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.disable_thinking and include_chat_template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

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
    def __init__(self, *args: Any, compact_prompt: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compact_prompt = bool(compact_prompt)

    def _summary_evidence_line(self, idx: int, evidence: Dict[str, object]) -> str:
        base = (
            f"{idx}. evidence_id={idx}; "
            f"hist_pid={evidence.get('problem_id', '')}; "
            f"hist_semantic_id={evidence.get('semantic_id', '')}; "
            f"history_pos={evidence.get('history_pos', '')}; "
            f"role={evidence.get('role', '')}; "
            f"overlap={evidence.get('knowledge_overlap', '')}; "
            f"level_diff={evidence.get('level_diff', '')}; "
            f"answer={evidence.get('answer_result', '')}; "
            f"support_score={evidence.get('support_score', '')}; "
        )
        if not self.compact_prompt:
            raw_scores = evidence.get("raw_scores", {})
            activation = evidence.get("activation", {})
            base += (
                f"activation={json.dumps(activation, ensure_ascii=False)}; "
                f"raw_scores={json.dumps(raw_scores, ensure_ascii=False)}; "
            )
        return base + f"text={evidence.get('question_text', '')}"

    def build_summary_prompts(
        self,
        *,
        target_pid: str,
        target_question_text: str,
        target_semantic_id: str,
        target_concepts: Iterable[str],
        evidence_list: Iterable[Dict[str, object]],
        template_summary_text: str,
    ) -> Tuple[str, str]:
        evidence_lines: List[str] = []
        for idx, evidence in enumerate(evidence_list, start=1):
            evidence_lines.append(self._summary_evidence_line(idx, evidence))

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
            '{"mastered_concepts":[{"concept":"等差数列求公差","evidence_ids":[1,3],"confidence":"高"}],"weak_concepts":[{"concept":"等差数列求和","evidence_ids":[2],"confidence":"中"}],"transfer_state":"同质迁移稳定","risk_level":"中","evidence_quality":"高","diagnosis":"等差数列求公差相关证据较稳定，但求和迁移仍有风险。"}\n\n'
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
        return system_prompt, prompt

    def prompt_signature(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        body = self._request_body(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            include_chat_template_kwargs=self._allow_chat_template_kwargs,
        )
        payload = {
            "schema_version": "strict_llm_summary_v1",
            "request_body": body,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        import hashlib

        return "prompt-signature\t" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def summarize_from_prompts(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        target_pid: str,
        target_semantic_id: str,
    ) -> str:
        current_user_prompt = user_prompt
        last_exc: Optional[Exception] = None
        last_raw_json = ""
        for attempt in range(3):
            try:
                obj = self.request_json(system_prompt=system_prompt, user_prompt=current_user_prompt)
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    break
                current_user_prompt = self._strict_summary_regeneration_prompt(
                    original_user_prompt=user_prompt,
                    error=exc,
                    invalid_json="",
                )
                continue
            raw_json = json.dumps(obj, ensure_ascii=False)
            last_raw_json = raw_json
            try:
                parsed = parse_llm_summary_json(raw_json)
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    break
                current_user_prompt = self._strict_summary_regeneration_prompt(
                    original_user_prompt=user_prompt,
                    error=exc,
                    invalid_json=raw_json,
                )

        hard_schema_prompt = (
            "请直接填写下面这个 JSON 模板，不得增加或删除字段：\n"
            "{\"mastered_concepts\":[],\"weak_concepts\":[],\"transfer_state\":\"证据有限\",\"risk_level\":\"中\",\"evidence_quality\":\"低\",\"diagnosis\":\"当前证据有限，稳定掌握点与薄弱点暂不明显。\"}\n\n"
            "只能依据输入证据填写；concept 只能来自目标题知识点或 overlap；evidence_ids 必须引用证据编号。\n\n"
            + user_prompt
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

    @staticmethod
    def _strict_summary_regeneration_prompt(
        *,
        original_user_prompt: str,
        error: Exception,
        invalid_json: str = "",
    ) -> str:
        invalid_block = f"\n上一次非法 JSON:\n{invalid_json}\n" if str(invalid_json or "").strip() else ""
        return (
            "上一次输出不是可解析的合法 JSON。请重新生成，不要修补原字符串。\n"
            "你必须只输出一个 JSON object，不能输出 Markdown、解释、代码块或 <think>。\n"
            "顶层字段必须且只能是：mastered_concepts, weak_concepts, transfer_state, risk_level, evidence_quality, diagnosis。\n"
            "mastered_concepts 和 weak_concepts 必须是数组；每个元素必须是 "
            "{\"concept\":\"...\",\"evidence_ids\":[1],\"confidence\":\"低|中|高\"}。\n"
            "risk_level 和 evidence_quality 只能是 \"低\"、\"中\"、\"高\"。\n"
            "evidence_ids 只能引用输入证据编号；没有可靠证据时用空数组。\n"
            "diagnosis 必须是 1 句中文，长度不超过 60 字。\n"
            "不要在字符串里使用未转义的双引号；如果需要引用题干内容，请改写而不是复制。\n"
            f"错误原因: {type(error).__name__}: {error}\n"
            f"{invalid_block}\n"
            "原始任务如下：\n"
            f"{original_user_prompt}"
        )

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
        system_prompt, user_prompt = self.build_summary_prompts(
            target_pid=target_pid,
            target_question_text=target_question_text,
            target_semantic_id=target_semantic_id,
            target_concepts=target_concepts,
            evidence_list=evidence_list,
            template_summary_text=template_summary_text,
        )
        return self.summarize_from_prompts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            target_pid=target_pid,
            target_semantic_id=target_semantic_id,
        )

    def _summary_evidence_lines(self, evidence_list: Iterable[Dict[str, object]]) -> List[str]:
        evidence_lines: List[str] = []
        for idx, evidence in enumerate(evidence_list, start=1):
            evidence_lines.append(self._summary_evidence_line(idx, evidence))
        return evidence_lines

    def _summary_case_block(self, case: Dict[str, object]) -> str:
        evidence_lines = self._summary_evidence_lines(case.get("evidence_list", []) or [])
        target_concepts = case.get("target_concepts", []) or []
        return (
            f"案例ID: {case.get('case_id', '')}\n"
            f"目标题ID: {case.get('target_pid', '')}\n"
            f"目标题语义ID: {case.get('target_semantic_id', '')}\n"
            f"目标题文本: {case.get('target_question_text', '')}\n"
            f"目标题知识点: {'、'.join(str(x) for x in target_concepts)}\n"
            f"学生近期模板统计摘要: {case.get('template_summary_text', '')}\n\n"
            "认知证据:\n"
            + "\n".join(evidence_lines)
        )

    def build_batch_summary_prompts(self, *, cases: Iterable[Dict[str, object]]) -> Tuple[str, str]:
        case_list = list(cases)
        if not case_list:
            raise ValueError("batch summary cases is empty")
        case_blocks = []
        case_ids: List[str] = []
        for case in case_list:
            case_id = str(case.get("case_id") or "").strip()
            if not case_id:
                raise ValueError("batch summary case_id is empty")
            case_ids.append(case_id)
            case_blocks.append("[CASE]\n" + self._summary_case_block(case).strip())

        system_prompt = (
            "你只输出一个合法 JSON 对象。"
            "顶层必须且只能包含 items 字段。"
            "items 中每个元素必须包含 case_id 和 summary。"
            "summary 必须同时包含 mastered_concepts、weak_concepts、transfer_state、risk_level、evidence_quality、diagnosis 这 6 个字段。"
            "不得输出 <think>、解释、分析过程、Markdown 代码块或额外文本。"
        )
        prompt = (
            "你是一个教育认知诊断压缩器。\n\n"
            "任务：对下面每个案例分别生成结构化诊断 JSON。每个案例必须独立判断，不能把不同案例的证据相互混用。\n\n"
            "严格输出格式：\n"
            "1. 只能输出一个 JSON 对象，顶层必须且只能是 {\"items\":[...]}。\n"
            "2. items 长度必须等于案例数量，且必须覆盖所有 case_id。\n"
            "3. 每个 item 必须且只能包含 case_id 和 summary。\n"
            "4. 每个 summary 必须且只能包含 mastered_concepts, weak_concepts, transfer_state, risk_level, evidence_quality, diagnosis 这 6 个字段。\n"
            "5. mastered_concepts 和 weak_concepts 必须是数组，最多各 3 项；没有内容时输出 []。\n"
            "6. mastered_concepts/weak_concepts 的每一项必须是对象：{\"concept\":字符串,\"evidence_ids\":整数数组,\"confidence\":\"低/中/高\"}。\n"
            "7. concept 只能来自对应案例的目标题知识点或证据 overlap，不得编造输入中不存在的知识点。\n"
            "8. evidence_ids 必须引用对应案例输入证据里的 evidence_id；若证据不足可为空数组。\n"
            "9. risk_level 和 evidence_quality 只能取 低、中、高。\n"
            "10. transfer_state 用一个短语概括迁移状态，例如：同质迁移稳定、前置不足、高阶迁移风险、协同证据不足、证据有限。\n"
            "11. diagnosis 必须是 1 到 2 句中文，总长度不超过 80 字，且不能为空。\n"
            "12. 如果证据冲突、主要依赖协同、或知识重合弱，应降低 evidence_quality 或 confidence。\n\n"
            "合法输出示例：\n"
            '{"items":[{"case_id":"case_0","summary":{"mastered_concepts":[{"concept":"等差数列求公差","evidence_ids":[1,3],"confidence":"高"}],"weak_concepts":[{"concept":"等差数列求和","evidence_ids":[2],"confidence":"中"}],"transfer_state":"同质迁移稳定","risk_level":"中","evidence_quality":"高","diagnosis":"等差数列求公差相关证据较稳定，但求和迁移仍有风险。"}}]}\n\n'
            f"必须覆盖的 case_id: {json.dumps(case_ids, ensure_ascii=False)}\n\n"
            "案例列表:\n"
            + "\n\n".join(case_blocks)
        )
        return system_prompt, prompt

    @staticmethod
    def _parse_batch_summary_obj(obj: Dict[str, Any], expected_case_ids: Iterable[str]) -> Dict[str, str]:
        expected = [str(item) for item in expected_case_ids]
        expected_set = set(expected)
        if set(obj.keys()) != {"items"}:
            raise ValueError(f"Unexpected batch summary top-level fields: {sorted(obj.keys())}")
        items = obj.get("items")
        if not isinstance(items, list):
            raise ValueError("batch summary items is not a list")
        if len(items) != len(expected):
            raise ValueError(f"batch summary item count mismatch: got {len(items)}, expected {len(expected)}")
        result: Dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("batch summary item is not an object")
            if set(item.keys()) != {"case_id", "summary"}:
                raise ValueError(f"Unexpected batch summary item fields: {sorted(item.keys())}")
            case_id = str(item.get("case_id") or "").strip()
            if case_id not in expected_set:
                raise ValueError(f"Unexpected batch summary case_id: {case_id}")
            if case_id in result:
                raise ValueError(f"Duplicate batch summary case_id: {case_id}")
            summary_obj = item.get("summary")
            if isinstance(summary_obj, str):
                parsed = parse_llm_summary_json(summary_obj)
            elif isinstance(summary_obj, dict):
                parsed = parse_llm_summary_json(json.dumps(summary_obj, ensure_ascii=False))
            else:
                raise ValueError(f"Invalid summary payload for case_id={case_id}")
            result[case_id] = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        missing = expected_set - set(result)
        if missing:
            raise ValueError(f"Missing batch summary case_id values: {sorted(missing)}")
        return result

    def summarize_batch(self, *, cases: Iterable[Dict[str, object]]) -> Dict[str, str]:
        case_list = list(cases)
        case_ids = [str(case.get("case_id") or "").strip() for case in case_list]
        system_prompt, user_prompt = self.build_batch_summary_prompts(cases=case_list)
        current_user_prompt = user_prompt
        last_exc: Optional[Exception] = None
        last_raw_json = ""
        for attempt in range(3):
            try:
                obj = self.request_json(system_prompt=system_prompt, user_prompt=current_user_prompt)
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    break
                current_user_prompt = (
                    "上一次输出的批量 JSON 不能解析。请重新生成，不要修补原字符串。\n"
                    "只能输出一个 JSON object，顶层必须且只能包含 items。\n"
                    "items 必须覆盖所有 case_id；每个 item 只能包含 case_id 和 summary。\n"
                    "每个 summary 必须只包含 mastered_concepts, weak_concepts, transfer_state, risk_level, evidence_quality, diagnosis 这 6 个字段。\n"
                    "risk_level/evidence_quality/confidence 只能使用 低、中、高。\n"
                    f"必须覆盖的 case_id: {json.dumps(case_ids, ensure_ascii=False)}\n"
                    f"错误原因: {type(exc).__name__}: {exc}\n\n"
                    "原始任务如下：\n"
                    f"{user_prompt}"
                )
                continue
            raw_json = json.dumps(obj, ensure_ascii=False)
            last_raw_json = raw_json
            try:
                return self._parse_batch_summary_obj(obj, case_ids)
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    break
                current_user_prompt = (
                    "你上一次输出的批量 JSON 不合法，请严格修复。\n"
                    "顶层必须只包含 items；items 必须覆盖所有 case_id；"
                    "每个 item 只能包含 case_id 和 summary；"
                    "每个 summary 必须只包含 mastered_concepts, weak_concepts, transfer_state, risk_level, evidence_quality, diagnosis 这 6 个字段。\n\n"
                    f"必须覆盖的 case_id: {json.dumps(case_ids, ensure_ascii=False)}\n\n"
                    f"原始非法 JSON:\n{raw_json}\n\n"
                    f"错误原因: {exc}"
                )
        raise ValueError(f"Invalid batch LLM summaries: {last_exc}. raw_obj={last_raw_json}") from last_exc


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
