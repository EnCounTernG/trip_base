"""工具模块。

本模块提供 Phase A 的最小工具集：
- 天气查询（Mock）
- 预算估算

并通过 LangChain 的 @tool 装饰器暴露为可调用工具。
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool


@tool
def weather_tool(city: str) -> str:
    """查询城市天气（Mock 版本，便于本地直接运行）。"""
    weather_map = {
        "上海": "多云，18~24℃，适合户外+室内混合行程",
        "北京": "晴，12~20℃，早晚温差较大",
        "杭州": "小雨，16~22℃，建议增加室内备选",
    }
    return weather_map.get(city, f"{city}：天气数据暂不可用，建议准备晴雨两套方案")


@tool
def budget_tool(days: int, travelers: int, level: str = "standard") -> str:
    """根据天数、人数、档次给出粗略预算区间。"""
    base = {"budget": 350, "standard": 650, "comfort": 1000}.get(level, 650)
    total = base * days * travelers
    low = int(total * 0.85)
    high = int(total * 1.15)
    return f"预算估算：{days}天/{travelers}人/{level}档，约 {low}~{high} 元"


def get_tools() -> Dict[str, Any]:
    """返回工具注册表，供执行节点按名称调用。"""
    return {
        "weather_tool": weather_tool,
        "budget_tool": budget_tool,
    }
