"""
工业日志分析器 - Web 界面 (RAG 增强版 v3.0)
新增：多格式日志兼容、API 重试、告警去重、趋势图、导出报告、文本粘贴输入、知识库健康检查
"""

import sys
import os
import re
import io
import json
import time
from datetime import datetime
from collections import defaultdict

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# RAG 模块（优雅降级）
try:
    import rag as rag_module
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# 文档解析可选依赖
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
load_dotenv()

# ── 常量 ──────────────────────────────────────────────────────────────────────

API_MAX_RETRIES = 3
API_RETRY_DELAY = 2  # 重试间隔（秒）

# 多格式日志正则，按优先级排列
LOG_PATTERNS = [
    # 标准格式：2024-01-15 08:32:11 [ERROR] PLC-01 message
    re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s+\[(?P<level>\w+)\]\s+(?P<device>\S+)\s+(?P<message>.+)"
    ),
    # 无括号格式：2024-01-15 08:32:11 ERROR PLC-01 message
    re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s+(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)"
        r"\s+(?P<device>\S+)\s+(?P<message>.+)"
    ),
    # ISO 8601：2024-01-15T08:32:11 [ERROR] PLC-01 message
    re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
        r"\s+\[(?P<level>\w+)\]\s+(?P<device>\S+)\s+(?P<message>.+)"
    ),
    # syslog 风格：Jan 15 08:32:11 PLC-01 ERROR: message
    re.compile(
        r"(?P<timestamp>[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
        r"\s+(?P<device>\S+)\s+(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)"
        r"[:\s]+(?P<message>.+)"
    ),
]

TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%b %d %H:%M:%S",
    "%b  %d %H:%M:%S",
]

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _parse_timestamp(ts_str: str) -> datetime | None:
    """尝试多种格式解析时间字符串，失败返回 None。"""
    for fmt in TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(ts_str.strip(), fmt)
            if dt.year == 1900:  # syslog 无年份，补当前年
                dt = dt.replace(year=datetime.now().year)
            return dt
        except ValueError:
            continue
    return None


def _try_parse_json(line: str, filename: str, lineno: int) -> dict | None:
    """尝试将行解析为 JSON 日志格式，支持常见字段名变体。"""
    try:
        obj = json.loads(line)
        level = (obj.get("level") or obj.get("severity") or obj.get("lvl") or "INFO").upper()
        device = str(obj.get("device") or obj.get("host") or obj.get("logger") or "UNKNOWN")
        message = str(obj.get("message") or obj.get("msg") or obj.get("text") or line)
        ts_raw = obj.get("timestamp") or obj.get("time") or obj.get("datetime") or ""
        ts = _parse_timestamp(str(ts_raw)) if ts_raw else datetime.now()
        return {
            "timestamp": ts,
            "level": level,
            "device": device,
            "message": message,
            "source_file": filename,
            "lineno": lineno,
        }
    except (json.JSONDecodeError, Exception):
        return None


def parse_log_text(text: str, filename: str) -> tuple[list[dict], int]:
    """
    解析日志文本，兼容标准格式、无括号格式、ISO 8601、syslog、JSON 行日志。
    返回 (结构化条目列表, 无法识别的行数)。
    """
    entries, skipped = [], 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        matched = False

        # 尝试 JSON 格式
        if line.startswith("{"):
            entry = _try_parse_json(line, filename, lineno)
            if entry:
                entries.append(entry)
                matched = True

        # 尝试各种正则格式
        if not matched:
            for pattern in LOG_PATTERNS:
                m = pattern.match(line)
                if m:
                    entry = m.groupdict()
                    ts = _parse_timestamp(entry["timestamp"])
                    if ts is None:
                        continue
                    entry["timestamp"] = ts
                    entry["level"] = (entry["level"].upper()
                                      .replace("WARN", "WARNING")
                                      .replace("FATAL", "CRITICAL"))
                    entry["source_file"] = filename
                    entry["lineno"] = lineno
                    entries.append(entry)
                    matched = True
                    break

        if not matched:
            skipped += 1

    return entries, skipped


def _normalize_message(msg: str) -> str:
    """将消息中的数字替换为 # 用于去重比较。"""
    return re.sub(r"[\d.]+", "#", msg).strip().lower()


def analyze(entries: list[dict]) -> dict:
    """
    统计日志条目，提取告警，并对重复告警去重。
    去重规则：同一设备 + 相同消息模式（忽略数值差异）视为同类告警，保留最新一条并记录出现次数。
    """
    stats = {
        "total": len(entries),
        "by_level": defaultdict(int),
        "by_device": defaultdict(int),
        "by_hour": defaultdict(int),
        "errors": [],
        "deduped_errors": [],
    }
    for e in entries:
        stats["by_level"][e["level"]] += 1
        stats["by_device"][e["device"]] += 1
        if e["level"] in ("ERROR", "CRITICAL"):
            hour_key = e["timestamp"].strftime("%m-%d %H:00")
            stats["by_hour"][hour_key] += 1
            stats["errors"].append(e)

    # 告警去重
    seen: dict[tuple, dict] = {}
    for err in stats["errors"]:
        key = (err["device"], _normalize_message(err["message"]))
        if key not in seen:
            seen[key] = {**err, "count": 1}
        else:
            seen[key]["count"] += 1
            if err["timestamp"] > seen[key]["timestamp"]:
                seen[key].update({
                    "timestamp": err["timestamp"],
                    "lineno": err["lineno"],
                    "source_file": err["source_file"],
                    "message": err["message"],
                })
    stats["deduped_errors"] = sorted(seen.values(), key=lambda x: x["timestamp"], reverse=True)
    return stats


def call_deepseek_api(entry: dict, use_rag: bool = True) -> tuple[str, list[dict]]:
    """
    调用 DeepSeek 生成中文诊断建议，网络异常时自动重试最多 3 次。
    返回 (建议文字, 检索到的知识片段列表)。
    """
    context_chunks: list[dict] = []
    if use_rag and RAG_AVAILABLE:
        try:
            context_chunks = rag_module.retrieve(f"{entry['device']} {entry['message']}", k=3)
        except Exception:
            context_chunks = []

    system_msg = (
        "你是一名工业设备运维专家，擅长诊断设备告警并给出处理建议。"
        "若无法根据现有信息判断原因，请如实说明'暂无足够信息判断，建议人工排查'，"
        "不要凭空猜测或捏造处理步骤。"
    )

    rag_section = ""
    if context_chunks:
        rag_section = "\n\n【参考知识库（仅参考相关度达标的片段）】\n" + "\n---\n".join(
            f"[来源: {c['source']}，相关度 {c['score']:.2f}]\n{c['content']}"
            for c in context_chunks
        )

    user_msg = (
        f"工业设备告警：\n"
        f"  设备：{entry['device']}\n"
        f"  时间：{entry['timestamp']}\n"
        f"  信息：{entry['message'][:500]}"
        + rag_section
        + "\n\n请结合以上信息给出一句通俗易懂的中文处理建议。只输出建议语句，不要任何额外说明。"
    )

    last_error = None
    for attempt in range(API_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=200,
                temperature=0.3,
                timeout=30,
            )
            return response.choices[0].message.content.strip(), context_chunks
        except Exception as e:
            last_error = e
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_DELAY)

    return f"⚠️ AI 建议获取失败（已重试 {API_MAX_RETRIES} 次）：{last_error}", context_chunks


def extract_text(uploaded_file) -> str:
    """从上传文件提取纯文本，支持 TXT / MD / PDF / DOCX，解析失败时返回空字符串。"""
    name = uploaded_file.name.lower()
    try:
        raw = uploaded_file.read()
        if name.endswith((".txt", ".md", ".log")):
            return raw.decode("utf-8", errors="replace")
        if name.endswith(".pdf"):
            if not PDF_OK:
                st.error("请先安装 pdfplumber：pip install pdfplumber")
                return ""
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
            if not text:
                st.warning(f"⚠️ {uploaded_file.name} 未能提取到文字（可能是扫描版 PDF）。")
            return text
        if name.endswith(".docx"):
            if not DOCX_OK:
                st.error("请先安装 python-docx：pip install python-docx")
                return ""
            doc = DocxDocument(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        st.warning(f"⚠️ 不支持的文件格式：{uploaded_file.name}")
        return ""
    except Exception as e:
        st.error(f"❌ 读取 {uploaded_file.name} 时出错：{e}")
        return ""


def build_export_report(stats: dict, source_hint: str = "") -> str:
    """生成可下载的文本分析报告，包含 AI 建议（若已获取）。"""
    lines = [
        "=" * 52,
        "        工业日志分析报告",
        "=" * 52,
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if source_hint:
        lines.append(f"分析来源：{source_hint}")
    lines += [
        f"日志总条数：{stats['total']}",
        "",
        "--- 按事件级别统计 ---",
    ]
    for level, count in sorted(stats["by_level"].items()):
        lines.append(f"  {level:<12} {count} 条")
    lines += ["", "--- 按设备统计（从高到低）---"]
    for device, count in sorted(stats["by_device"].items(), key=lambda x: -x[1]):
        lines.append(f"  {device:<20} {count} 条")

    deduped = stats.get("deduped_errors", [])
    lines += ["", f"--- 错误与严重告警（去重后共 {len(deduped)} 类）---", ""]
    for err in deduped:
        count_tag = f"（共出现 {err['count']} 次）" if err.get("count", 1) > 1 else ""
        lines.append(f"  [{err['timestamp']}] {err['device']} | {err['level']}{count_tag}")
        lines.append(f"  信息：{err['message']}")
        lines.append(f"  来源：{err['source_file']} 第 {err['lineno']} 行")
        if err.get("advice"):
            lines.append(f"  >>> AI 建议：{err['advice']}")
        lines.append("")
    lines.append("=" * 52)
    return "\n".join(lines)


def render_trend_chart(stats: dict):
    """绘制每小时 ERROR/CRITICAL 告警数量柱状图。"""
    import pandas as pd
    data = {"时间": [], "告警数": []}
    for hour, count in sorted(stats["by_hour"].items()):
        data["时间"].append(hour)
        data["告警数"].append(count)
    df = pd.DataFrame(data).set_index("时间")
    st.bar_chart(df, color="#FF4B4B")


# ── 页面配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="工业日志分析器", page_icon="🏭", layout="wide")
st.title("🏭 工业日志分析器（RAG 增强版 v3.0）")

if not api_key:
    st.error("未检测到 OPENAI_API_KEY，请在 .env 文件中配置后重启。")
    st.stop()

# ── 侧边栏：知识库健康检查 ──────────────────────────────────────────────────
if RAG_AVAILABLE:
    try:
        health = rag_module.health_check()
        if health["ok"]:
            st.sidebar.success(f"✅ 知识库正常（{health['count']} 个片段）")
        else:
            st.sidebar.error(f"❌ 知识库异常：{health['error']}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ 知识库状态未知：{e}")
else:
    st.sidebar.warning("RAG 依赖未安装，知识库功能已禁用。")

tab_log, tab_kb = st.tabs(["📋 日志分析", "📚 知识库管理"])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — 日志分析
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    # ── 控制选项 ──
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        use_rag = RAG_AVAILABLE and st.toggle(
            "启用 RAG 知识库增强", value=True,
            help="开启后分析告警时自动检索知识库中的相关工单/文档"
        )
    with ctrl_col2:
        use_dedup = st.toggle(
            "启用告警去重", value=True,
            help="将同设备同类型的重复告警合并为一条，并显示出现次数"
        )

    # ── 输入区：上传文件 / 粘贴文本 ──
    st.subheader("📂 输入日志")
    input_tab1, input_tab2 = st.tabs(["📁 上传文件", "📋 粘贴文本"])

    all_entries: list[dict] = []
    total_skipped = 0
    source_names: list[str] = []
    uploaded_files = []
    pasted_text = ""

    with input_tab1:
        uploaded_files = st.file_uploader(
            "支持同时上传多个文件（.log / .txt），兼容标准格式、syslog、JSON 行日志",
            type=["log", "txt"],
            accept_multiple_files=True,
            key="log_uploader",
        )
        if uploaded_files:
            for f in uploaded_files:
                text = f.read().decode("utf-8", errors="replace")
                entries, skipped = parse_log_text(text, f.name)
                all_entries.extend(entries)
                total_skipped += skipped
                source_names.append(f.name)

    with input_tab2:
        pasted_text = st.text_area(
            "将日志内容粘贴到此处",
            height=200,
            placeholder=(
                "2024-01-15 08:20:33 [ERROR] PLC-01 Temperature sensor fault: value=102.3\n"
                "2024-01-15 08:40:00 [CRITICAL] MOTOR-01 Motor stall detected\n"
                "或 JSON 格式：{\"timestamp\":\"2024-01-15 08:20:33\",\"level\":\"ERROR\",...}"
            ),
        )
        paste_name = st.text_input("来源标记（用于报告）", value="pasted_input.log")
        if pasted_text.strip():
            entries, skipped = parse_log_text(pasted_text, paste_name)
            all_entries.extend(entries)
            total_skipped += skipped
            source_names.append(paste_name)

    # ── 分析区 ──
    has_input = bool(uploaded_files) or bool(pasted_text.strip())
    if not has_input:
        st.info("请上传 .log 文件或在「粘贴文本」Tab 中输入日志内容。")
    elif not all_entries:
        st.warning(
            "⚠️ 未识别到任何有效日志条目，请确认格式符合以下之一：\n\n"
            "- 标准：`2024-01-15 08:32:11 [ERROR] PLC-01 message`\n"
            "- 无括号：`2024-01-15 08:32:11 ERROR PLC-01 message`\n"
            "- ISO 8601：`2024-01-15T08:32:11 [ERROR] PLC-01 message`\n"
            "- syslog：`Jan 15 08:32:11 PLC-01 ERROR: message`\n"
            "- JSON 行：`{\"timestamp\":\"...\",\"level\":\"ERROR\",\"device\":\"PLC-01\",\"message\":\"...\"}`"
        )
    else:
        if total_skipped > 0:
            st.info(f"ℹ️ {total_skipped} 行因格式不匹配被跳过，不影响已识别条目的分析。")

        stats = analyze(all_entries)
        errors_to_show = stats["deduped_errors"] if use_dedup else stats["errors"]

        # ── 统计概览 ──
        st.subheader("📊 统计概览")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("日志总条数", stats["total"])
        c2.metric("CRITICAL", stats["by_level"].get("CRITICAL", 0), delta_color="inverse")
        c3.metric("ERROR", stats["by_level"].get("ERROR", 0), delta_color="inverse")
        c4.metric("WARNING", stats["by_level"].get("WARNING", 0), delta_color="inverse")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**按事件级别统计**")
            st.table([{"级别": k, "条数": v} for k, v in sorted(stats["by_level"].items())])
        with col_r:
            st.markdown("**按设备统计（从高到低）**")
            st.table([
                {"设备": k, "条数": v}
                for k, v in sorted(stats["by_device"].items(), key=lambda x: -x[1])
            ])

        # ── 趋势图 ──
        if stats["by_hour"]:
            st.subheader("📈 ERROR/CRITICAL 告警趋势（按小时）")
            render_trend_chart(stats)

        # ── 告警列表 ──
        dedup_note = (
            f"，去重后 {len(errors_to_show)} 类"
            if use_dedup and len(errors_to_show) < len(stats["errors"])
            else ""
        )
        st.subheader(f"⚠️ 错误与严重告警（共 {len(stats['errors'])} 条{dedup_note}）")

        if not errors_to_show:
            st.success("未发现任何 ERROR 或 CRITICAL 级别的告警。")
        else:
            if st.button("🤖 一键获取 DeepSeek 中文建议", type="primary"):
                for i, err in enumerate(errors_to_show):
                    with st.spinner(f"正在分析第 {i+1}/{len(errors_to_show)} 条..."):
                        advice, chunks = call_deepseek_api(err, use_rag=use_rag)
                    err["advice"] = advice  # 存入 err 供导出使用

                    color = "🔴" if err["level"] == "CRITICAL" else "🟠"
                    with st.container(border=True):
                        count_badge = f"　**×{err['count']} 次重复**" if err.get("count", 1) > 1 else ""
                        st.markdown(
                            f"{color} **[{err['timestamp']}] {err['device']}**　"
                            f"`{err['level']}`{count_badge}"
                        )
                        st.markdown(f"**告警信息：** {err['message']}")
                        st.markdown(f"**来源：** {err['source_file']} 第 {err['lineno']} 行")
                        st.success(f"💡 中文建议：{advice}")
                        if chunks:
                            with st.expander(f"📖 参考了 {len(chunks)} 条知识库片段"):
                                for c in chunks:
                                    st.caption(f"[{c['source']}]　相似度 {c['score']:.2f}")
                                    st.markdown(
                                        f"> {c['content'][:200]}"
                                        f"{'...' if len(c['content']) > 200 else ''}"
                                    )

                # ── 导出报告 ──
                st.divider()
                report_text = build_export_report(stats, source_hint=", ".join(source_names))
                st.download_button(
                    label="📥 下载分析报告（TXT）",
                    data=report_text.encode("utf-8"),
                    file_name=f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

            else:
                for err in errors_to_show:
                    color = "🔴" if err["level"] == "CRITICAL" else "🟠"
                    with st.container(border=True):
                        count_badge = f"　**×{err['count']} 次重复**" if err.get("count", 1) > 1 else ""
                        st.markdown(
                            f"{color} **[{err['timestamp']}] {err['device']}**　"
                            f"`{err['level']}`{count_badge}"
                        )
                        st.markdown(f"**告警信息：** {err['message']}")
                        st.markdown(f"**来源：** {err['source_file']} 第 {err['lineno']} 行")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — 知识库管理
# ══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    if not RAG_AVAILABLE:
        st.error(
            "RAG 依赖未安装，无法使用知识库功能。请执行：\n\n"
            "```\npip install chromadb sentence-transformers pdfplumber python-docx\n```"
        )
    else:
        st.subheader("📤 上传工单 / 技术文档")
        st.caption("支持 TXT、MD、PDF、DOCX 格式；同名文档重新上传将自动覆盖旧版本。")

        kb_files = st.file_uploader(
            "选择文件（可多选）",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            key="kb_uploader",
        )

        if kb_files:
            if st.button("⬆️ 入库", type="primary"):
                for f in kb_files:
                    text = extract_text(f)
                    if text.strip():
                        try:
                            with st.spinner(f"正在嵌入 {f.name}（首次运行需下载嵌入模型，请耐心等待）..."):
                                n = rag_module.add_document(f.name, text)
                            st.success(f"✅ {f.name} 已入库（{n} 个片段）")
                        except Exception as e:
                            st.error(f"❌ {f.name} 入库失败：{e}")
                    else:
                        st.warning(f"⚠️ {f.name} 内容为空，已跳过。")
                st.rerun()

        st.divider()
        st.subheader("📑 当前知识库文档")

        docs = rag_module.list_documents()
        if not docs:
            st.info("知识库为空，请先上传文档。")
        else:
            st.caption(f"共 {len(docs)} 篇文档，{sum(d['chunks'] for d in docs)} 个片段")
            st.divider()
            for doc in docs:
                col_name, col_chunks, col_del = st.columns([5, 2, 1])
                col_name.markdown(f"**{doc['filename']}**")
                col_chunks.caption(f"{doc['chunks']} 个片段")
                if col_del.button("🗑️ 删除", key=f"del_{doc['filename']}"):
                    n = rag_module.delete_document(doc["filename"])
                    st.toast(f"已删除 {doc['filename']}（{n} 个片段）", icon="✅")
                    st.rerun()
