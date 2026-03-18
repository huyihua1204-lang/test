"""
工业日志分析器
从工业设备日志中提取告警信息，调用 DeepSeek 生成中文诊断建议，并输出分析报告。
"""

import sys
import os
import re
import logging
from pathlib import Path
from datetime import datetime

# 修复 Windows 控制台中文乱码，强制使用 UTF-8 输出
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv


# ──────────────────────────────────────────────
# 日志格式正则：匹配形如
# "2024-01-15 08:32:11 [ERROR] PLC-01 Temperature sensor fault: value=102.3"
# ──────────────────────────────────────────────
LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"\s+\[(?P<level>\w+)\]"
    r"\s+(?P<device>\S+)"
    r"\s+(?P<message>.+)"
)


# ══════════════════════════════════════════════
# 模块一：load_env_config()
# 职责：安全加载 .env 文件中的环境变量，
#       初始化日志系统与 DeepSeek 客户端，
#       返回配置字典供其他模块使用。
# ══════════════════════════════════════════════
def load_env_config() -> dict:
    """
    安全加载环境变量，初始化日志与 API 客户端。

    返回值：
        config (dict) 包含以下键：
            - log_dir     : 日志文件目录路径
            - output_dir  : 报告输出目录路径
            - client      : 已配置好的 DeepSeek OpenAI 客户端
            - logger      : 全局日志记录器
    """
    # 从 .env 文件加载环境变量（若变量已存在则不覆盖）
    load_dotenv()

    # 读取各项配置，提供默认值以防 .env 中未定义
    log_dir    = os.getenv("LOG_DIR",    "./logs")
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    log_level  = os.getenv("LOG_LEVEL",  "INFO")
    api_key    = os.getenv("OPENAI_API_KEY")

    # 校验 API Key 是否存在，缺失时提前报错，避免运行到一半才崩溃
    if not api_key:
        raise EnvironmentError("未找到 OPENAI_API_KEY，请在 .env 文件中设置 DeepSeek API 密钥。")

    # 初始化全局日志系统
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False))],
    )
    logger = logging.getLogger(__name__)

    # 初始化 DeepSeek 客户端（兼容 OpenAI SDK，仅替换 base_url）
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    logger.info("环境变量加载成功，日志目录：%s，输出目录：%s", log_dir, output_dir)

    return {
        "log_dir":    log_dir,
        "output_dir": output_dir,
        "client":     client,
        "logger":     logger,
    }


# ══════════════════════════════════════════════
# 模块二：read_logs(path)
# 职责：遍历指定目录下所有 .log 文件，
#       逐行解析日志，返回结构化的条目列表
#       以及按级别/设备统计的汇总数据。
# ══════════════════════════════════════════════
def read_logs(path: str, logger: logging.Logger) -> dict:
    """
    读取指定目录下的所有 .log 文件并解析成结构化数据。

    参数：
        path   : 日志文件夹路径（字符串）
        logger : 日志记录器

    返回值：
        stats (dict) 包含：
            - total     : 总条目数
            - by_level  : 各级别计数 {级别: 数量}
            - by_device : 各设备计数 {设备: 数量}
            - errors    : ERROR 和 CRITICAL 级别的条目列表
    """
    log_dir = Path(path)

    # 检查目录是否存在
    if not log_dir.exists():
        logger.error("日志目录不存在：%s", log_dir)
        raise FileNotFoundError(f"日志目录不存在：{log_dir}")

    # 收集所有 .log 文件
    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        logger.warning("目录 %s 中未找到任何 .log 文件", log_dir)
        raise FileNotFoundError(f"目录 {log_dir} 中无 .log 文件")

    logger.info("发现 %d 个日志文件，开始解析...", len(log_files))

    # 汇总统计容器
    stats = {
        "total":     0,
        "by_level":  defaultdict(int),
        "by_device": defaultdict(int),
        "errors":    [],
    }

    for filepath in log_files:
        entries = _parse_single_file(filepath, logger)
        logger.info("  %s → 解析到 %d 条记录", filepath.name, len(entries))

        # 统计每条记录
        for entry in entries:
            stats["total"] += 1
            stats["by_level"][entry["level"]] += 1
            stats["by_device"][entry["device"]] += 1

            # 仅保留 ERROR / CRITICAL 供后续 AI 分析
            if entry["level"] in ("ERROR", "CRITICAL"):
                stats["errors"].append(entry)

    logger.info("日志解析完成，共 %d 条，其中告警 %d 条", stats["total"], len(stats["errors"]))
    return stats


def _parse_single_file(filepath: Path, logger: logging.Logger) -> list[dict]:
    """（内部）解析单个日志文件，返回结构化条目列表。"""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            match = LOG_PATTERN.match(line)
            if match:
                entry = match.groupdict()
                # 将时间字符串转为 datetime 对象，便于后续格式化
                entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                entry["source_file"] = filepath.name
                entry["lineno"] = lineno
                entries.append(entry)
            else:
                logger.debug("跳过不匹配的行 %d（%s）", lineno, filepath.name)
    return entries


# ══════════════════════════════════════════════
# 模块三：call_deepseek_api(log_content)
# 职责：接收单条错误日志字典，
#       构造提示词并调用 DeepSeek API，
#       返回一句通俗易懂的中文处理建议。
# ══════════════════════════════════════════════
def call_deepseek_api(log_content: dict, client: OpenAI) -> str:
    """
    调用 DeepSeek API，针对单条错误日志生成中文诊断建议。

    参数：
        log_content : 单条日志条目字典，包含 device / timestamp / message 等字段
        client      : 已初始化的 DeepSeek OpenAI 客户端

    返回值：
        advice (str) : 一句中文建议，例如 'PLC-01 温度传感器过热，请立即检查散热模块。'
    """
    # 构造提示词：提供设备名、时间、告警信息，要求输出一句中文建议
    prompt = (
        f"工业设备告警：\n"
        f"  设备：{log_content['device']}\n"
        f"  时间：{log_content['timestamp']}\n"
        f"  信息：{log_content['message']}\n\n"
        "请用一句通俗易懂的中文给出处理建议，"
        "例如：'A2线电机过热，请立即检查散热系统。'\n"
        "只输出建议语句，不要任何额外说明。"
    )

    # 调用 DeepSeek Chat API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3,   # 降低随机性，使建议更稳定
    )

    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════
# 模块四：generate_report(results)
# 职责：接收 read_logs() 返回的统计数据，
#       对每条 ERROR/CRITICAL 调用 DeepSeek，
#       将完整分析结果写入 report.txt。
# ══════════════════════════════════════════════
def generate_report(results: dict, output_dir: str, client: OpenAI, logger: logging.Logger) -> Path:
    """
    将分析结果与 AI 中文建议写入 report.txt。

    参数：
        results    : read_logs() 返回的统计字典
        output_dir : 报告输出目录路径（字符串）
        client     : DeepSeek 客户端，用于逐条获取 AI 建议
        logger     : 日志记录器

    返回值：
        report_path (Path) : 生成的报告文件路径
    """
    output_path = Path(output_dir) / "report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        # ── 报告头部 ──
        f.write("=" * 50 + "\n")
        f.write("       工业日志分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"日志总条数：{results['total']}\n\n")

        # ── 按级别统计 ──
        f.write("--- 按事件级别统计 ---\n")
        for level, count in sorted(results["by_level"].items()):
            f.write(f"  {level:<10} {count} 条\n")

        # ── 按设备统计 ──
        f.write("\n--- 按设备统计（从高到低）---\n")
        for device, count in sorted(results["by_device"].items(), key=lambda x: -x[1]):
            f.write(f"  {device:<20} {count} 条\n")

        # ── 错误与严重告警（含 AI 中文建议）──
        f.write(f"\n--- 错误与严重告警（共 {len(results['errors'])} 条）---\n\n")
        for idx, err in enumerate(results["errors"], start=1):
            logger.info("正在为第 %d/%d 条告警请求 AI 建议...", idx, len(results["errors"]))

            # 调用模块三获取中文建议
            advice = call_deepseek_api(err, client)

            f.write(f"  [{err['timestamp']}] {err['device']} | {err['message']}\n")
            f.write(f"  来源：{err['source_file']} 第 {err['lineno']} 行\n")
            f.write(f"  >>> 中文建议：{advice}\n")
            f.write("\n")

    logger.info("报告已生成：%s", output_path)
    return output_path


# ══════════════════════════════════════════════
# 主入口：按顺序调用四个模块
# ══════════════════════════════════════════════
def main():
    # 第一步：加载配置与初始化
    config = load_env_config()
    logger = config["logger"]
    client = config["client"]

    # 第二步：读取并解析日志
    stats = read_logs(config["log_dir"], logger)

    # 第三步 & 第四步：生成报告（报告内部逐条调用 DeepSeek）
    report_path = generate_report(stats, config["output_dir"], client, logger)

    print(f"\n完成！共处理 {stats['total']} 条日志，"
          f"发现 {len(stats['errors'])} 条告警。\n报告路径：{report_path}")


if __name__ == "__main__":
    main()
