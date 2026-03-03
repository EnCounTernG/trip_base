"""应用状态定义模块。

本模块定义 LangGraph 在节点间流转的统一状态结构（AgentState）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph 运行时共享状态。

    字段说明：
    - user_query: 用户原始输入
    - intent: 意图分类结果（chat/plan/qa）
    - thoughts: ReAct 的思考轨迹（可观测）
    - current_action: 当前要执行的动作（工具调用）
    - tool_calls: 历史工具调用记录
    - observations: 工具返回结果记录
    - iteration: 当前 ReAct 迭代轮次
    - max_iterations: 最大迭代轮次，防止死循环
    - final_answer: 最终返回给用户的答案
    - error: 失败信息（可选）
    """

    user_query: str
    intent: Literal["chat", "plan", "qa"]
    thoughts: List[str]
    current_action: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    final_answer: str
    error: str
