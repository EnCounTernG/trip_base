"""LangGraph 工作流模块。

该模块实现 Phase A 的核心：
- Intent Router（意图路由）
- ReAct Planner（思考+决定动作）
- Tool Executor（执行工具）
- Answer Synthesizer（答案汇总）

说明：
- 这里采用“可本地直接运行”的 ReAct 极简实现。
- 为了教学可读性，保留思考轨迹 thoughts。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.state import AgentState
from app.tools import get_tools

# -----------------------------
# 1) LangChain Prompt + Runnable
# -----------------------------
# 使用 LangChain 的 PromptTemplate 与 Runnable 组合，构造一个“规划器链”。
# 这里不依赖外部大模型，确保你本地无 API Key 也能跑通 Phase A。
planner_prompt = ChatPromptTemplate.from_template(
    """
你是旅行助手的ReAct规划器，请根据用户问题与历史观察决定下一步。
用户问题: {user_query}
当前意图: {intent}
已执行轮次: {iteration}/{max_iterations}
历史观察: {observations}

你必须输出 JSON，格式如下：
{{
  "thought": "你的简短思考",
  "action": "weather_tool|budget_tool|final",
  "action_input": {{"key": "value"}},
  "final_answer": "当 action=final 时给出最终答案，否则留空"
}}
""".strip()
)


def _rule_based_planner(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """规则版规划器（模拟 LLM 规划结果）。

    作用：
    - 让 Phase A 可离线运行。
    - 通过 JSON 结构体现 ReAct 的 action 决策。
    """
    user_query = input_data["user_query"]
    iteration = input_data["iteration"]
    max_iterations = input_data["max_iterations"]

    if iteration >= max_iterations:
        return {
            "thought": "达到最大迭代次数，直接总结当前信息。",
            "action": "final",
            "action_input": {},
            "final_answer": "已达到最大规划轮次，建议先确认出行城市/天数/预算后我再继续细化。",
        }

    if "天气" in user_query or "下雨" in user_query:
        city_match = re.search(r"上海|北京|杭州", user_query)
        city = city_match.group(0) if city_match else "上海"
        return {
            "thought": "用户关注天气，先查天气再给建议。",
            "action": "weather_tool",
            "action_input": {"city": city},
            "final_answer": "",
        }

    if "预算" in user_query or "多少钱" in user_query:
        days_match = re.search(r"(\d+)天", user_query)
        people_match = re.search(r"(\d+)(?:人|位)", user_query)
        days = int(days_match.group(1)) if days_match else 3
        travelers = int(people_match.group(1)) if people_match else 2
        return {
            "thought": "用户关注预算，先进行预算估算。",
            "action": "budget_tool",
            "action_input": {"days": days, "travelers": travelers, "level": "standard"},
            "final_answer": "",
        }

    return {
        "thought": "信息已足够，直接输出行程建议。",
        "action": "final",
        "action_input": {},
        "final_answer": (
            "建议采用3天2晚经典行程：D1市区地标+D2博物馆与美食+D3休闲返程；"
            "如需我可继续按预算、亲子/情侣/老人同行进行细化。"
        ),
    }


planner_chain = planner_prompt | RunnableLambda(_rule_based_planner)


# -----------------------------
# 2) LangGraph 节点定义
# -----------------------------
def intent_router_node(state: AgentState) -> AgentState:
    """意图路由节点：将问题初步分类为 chat/plan/qa。"""
    query = state.get("user_query", "")
    if any(k in query for k in ["规划", "行程", "路线", "旅游", "旅行"]):
        intent: Literal["chat", "plan", "qa"] = "plan"
    elif any(k in query for k in ["是什么", "介绍", "知识", "规则"]):
        intent = "qa"
    else:
        intent = "chat"

    return {
        **state,
        "intent": intent,
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 4),
        "thoughts": state.get("thoughts", []),
        "tool_calls": state.get("tool_calls", []),
        "observations": state.get("observations", []),
    }


def planner_node(state: AgentState) -> AgentState:
    """ReAct 规划节点：生成 thought + action。"""
    result = planner_chain.invoke(
        {
            "user_query": state["user_query"],
            "intent": state["intent"],
            "iteration": state["iteration"],
            "max_iterations": state["max_iterations"],
            "observations": json.dumps(state.get("observations", []), ensure_ascii=False),
        }
    )

    thoughts = state.get("thoughts", []) + [result["thought"]]
    return {
        **state,
        "thoughts": thoughts,
        "current_action": {
            "name": result["action"],
            "input": result["action_input"],
            "final_answer": result.get("final_answer", ""),
        },
    }


def tool_executor_node(state: AgentState) -> AgentState:
    """工具执行节点：根据 current_action 调用工具并记录 observation。"""
    action = state.get("current_action", {})
    action_name = action.get("name", "final")

    # action=final 表示无需工具，直接进入答案汇总。
    if action_name == "final":
        return state

    tools = get_tools()
    tool_obj = tools.get(action_name)
    if tool_obj is None:
        return {**state, "error": f"未知工具: {action_name}"}

    tool_input = action.get("input", {})
    result = tool_obj.invoke(tool_input)

    tool_calls = state.get("tool_calls", []) + [{"tool": action_name, "input": tool_input}]
    observations = state.get("observations", []) + [{"tool": action_name, "result": result}]

    return {
        **state,
        "tool_calls": tool_calls,
        "observations": observations,
        "iteration": state.get("iteration", 0) + 1,
    }


def answer_synthesizer_node(state: AgentState) -> AgentState:
    """答案汇总节点：根据动作或观察构造最终回复。"""
    action = state.get("current_action", {})
    if action.get("name") == "final":
        answer = action.get("final_answer") or "我已经整理好了建议，如需可继续细化。"
    else:
        # 有 observation 时，基于工具结果生成用户可读答案。
        obs = state.get("observations", [])
        if obs:
            latest = obs[-1]
            answer = f"已完成工具查询：{latest['result']}。如果你愿意，我可以继续补全详细日程。"
        else:
            answer = "暂未获得足够信息，请补充城市、天数、人数、预算等信息。"

    return {**state, "final_answer": answer}


def should_continue(state: AgentState) -> Literal["tool", "end"]:
    """条件分支：planner 后决定进入工具执行还是结束。"""
    action = state.get("current_action", {})
    if action.get("name") == "final":
        return "end"
    return "tool"


# -----------------------------
# 3) 图构建与对外接口
# -----------------------------
def build_graph():
    """构建 LangGraph 编排图并返回可调用对象。"""
    graph = StateGraph(AgentState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("answer_synthesizer", answer_synthesizer_node)

    graph.add_edge(START, "intent_router")
    graph.add_edge("intent_router", "planner")
    graph.add_conditional_edges("planner", should_continue, {"tool": "tool_executor", "end": "answer_synthesizer"})
    graph.add_edge("tool_executor", "answer_synthesizer")
    graph.add_edge("answer_synthesizer", END)

    return graph.compile()


def run_agent(user_query: str, max_iterations: int = 4) -> Dict[str, Any]:
    """对外运行入口：输入用户问题，返回结构化结果。"""
    app = build_graph()
    init_state: AgentState = {
        "user_query": user_query,
        "max_iterations": max_iterations,
        "iteration": 0,
        "thoughts": [],
        "tool_calls": [],
        "observations": [],
    }
    result = app.invoke(init_state)
    return {
        "answer": result.get("final_answer", ""),
        "intent": result.get("intent", "chat"),
        "thoughts": result.get("thoughts", []),
        "tool_calls": result.get("tool_calls", []),
        "observations": result.get("observations", []),
        "error": result.get("error", ""),
    }
