"""基于LangGraph的旅行规划运行时。

阶段A目标:
1. 先完成执行框架迁移(HelloAgents -> LangGraph)
2. 保持前端接口不变,保障可平滑切换
3. 暂时沿用MCP能力获取地图信息
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..config import get_settings
from ..models.schemas import Attraction, DayPlan, Location, Meal, TripPlan, TripRequest
from ..services.amap_service import get_amap_mcp_tool
from ..services.langchain_service import get_langchain_llm


class TripPlanningState(TypedDict, total=False):
    """LangGraph状态对象。

    用于在各节点之间传递上下文与中间结果。
    """

    request: TripRequest
    attractions_raw: str
    weather_raw: str
    hotels_raw: str
    planner_raw: str
    trip_plan: TripPlan
    runtime: str
    errors: List[str]


class LangGraphTripPlanner:
    """基于LangGraph的旅行规划器。"""

    def __init__(self):
        settings = get_settings()
        self.runtime_name = "langgraph"
        self.llm = get_langchain_llm()

        # 复用当前项目已有的MCP工具能力,避免阶段A一次改动过大
        self.amap_tool = get_amap_mcp_tool()
        self.amap_api_key_configured = bool(settings.amap_api_key)

        self.graph = self._build_graph()

    def _build_graph(self):
        """构建LangGraph流程图。"""
        graph = StateGraph(TripPlanningState)

        graph.add_node("search_attractions", self._node_search_attractions)
        graph.add_node("query_weather", self._node_query_weather)
        graph.add_node("search_hotels", self._node_search_hotels)
        graph.add_node("plan_trip", self._node_plan_trip)

        # 阶段A先采用线性流程,便于和旧实现对齐
        graph.set_entry_point("search_attractions")
        graph.add_edge("search_attractions", "query_weather")
        graph.add_edge("query_weather", "search_hotels")
        graph.add_edge("search_hotels", "plan_trip")
        graph.add_edge("plan_trip", END)

        return graph.compile()

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """执行LangGraph流程并输出TripPlan。"""
        init_state: TripPlanningState = {
            "request": request,
            "runtime": self.runtime_name,
            "errors": [],
        }

        try:
            final_state = self.graph.invoke(init_state)
            if final_state.get("trip_plan"):
                return final_state["trip_plan"]

            return self._create_fallback_plan(request, "未生成结构化行程,使用降级方案")
        except Exception as exc:
            return self._create_fallback_plan(request, f"LangGraph执行异常: {exc}")

    def _call_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """统一封装MCP调用,便于阶段A快速验证。"""
        result = self.amap_tool.run(
            {
                "action": "call_tool",
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def _node_search_attractions(self, state: TripPlanningState) -> TripPlanningState:
        """节点1: 查询景点。"""
        request = state["request"]
        keyword = request.preferences[0] if request.preferences else "热门景点"

        try:
            raw = self._call_mcp(
                "maps_text_search",
                {"keywords": keyword, "city": request.city, "citylimit": "true"},
            )
        except Exception as exc:
            raw = f"景点检索失败: {exc}"
            state.setdefault("errors", []).append(raw)

        return {**state, "attractions_raw": raw}

    def _node_query_weather(self, state: TripPlanningState) -> TripPlanningState:
        """节点2: 查询天气。"""
        request = state["request"]

        try:
            raw = self._call_mcp("maps_weather", {"city": request.city})
        except Exception as exc:
            raw = f"天气检索失败: {exc}"
            state.setdefault("errors", []).append(raw)

        return {**state, "weather_raw": raw}

    def _node_search_hotels(self, state: TripPlanningState) -> TripPlanningState:
        """节点3: 查询酒店。"""
        request = state["request"]

        try:
            raw = self._call_mcp(
                "maps_text_search",
                {"keywords": f"{request.accommodation} 酒店", "city": request.city, "citylimit": "true"},
            )
        except Exception as exc:
            raw = f"酒店检索失败: {exc}"
            state.setdefault("errors", []).append(raw)

        return {**state, "hotels_raw": raw}

    def _node_plan_trip(self, state: TripPlanningState) -> TripPlanningState:
        """节点4: 用LangChain模型整合输出结构化行程。"""
        request = state["request"]

        prompt = f"""
你是旅行规划专家,请只返回JSON(不要markdown)。

输入信息:
- 城市: {request.city}
- 开始日期: {request.start_date}
- 结束日期: {request.end_date}
- 天数: {request.travel_days}
- 交通方式: {request.transportation}
- 住宿偏好: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}
- 用户额外要求: {request.free_text_input or '无'}

景点检索结果:
{state.get('attractions_raw', '')}

天气检索结果:
{state.get('weather_raw', '')}

酒店检索结果:
{state.get('hotels_raw', '')}

JSON字段要求:
- city,start_date,end_date,days,weather_info,overall_suggestions,budget
- days中必须有 date/day_index/description/transportation/accommodation/attractions/meals
- attractions中的location要包含longitude和latitude
"""

        try:
            message = self.llm.invoke(prompt)
            content = getattr(message, "content", "")
            plan = self._parse_plan(content, request)
            return {**state, "planner_raw": str(content), "trip_plan": plan}
        except Exception as exc:
            state.setdefault("errors", []).append(f"LLM规划失败: {exc}")
            plan = self._create_fallback_plan(request, f"LLM规划失败: {exc}")
            return {**state, "trip_plan": plan}

    def _parse_plan(self, content: Any, request: TripRequest) -> TripPlan:
        """将LLM输出解析为TripPlan。"""
        if not isinstance(content, str):
            return self._create_fallback_plan(request, "LLM输出非字符串")

        text = content.strip()

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif text.startswith("```"):
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
            return TripPlan(**data)
        except Exception:
            # 兜底: 尝试从首尾大括号截取
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                data = json.loads(text[start:end])
                return TripPlan(**data)
            except Exception:
                return self._create_fallback_plan(request, "JSON解析失败")

    def _create_fallback_plan(self, request: TripRequest, reason: str) -> TripPlan:
        """当图执行失败时,返回可用的降级行程。"""
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")

        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i + 1}天行程(降级方案)",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}推荐景点{i + 1}",
                        address=f"{request.city}市区",
                        location=Location(longitude=116.397128 + i * 0.01, latitude=39.916527 + i * 0.01),
                        visit_duration=120,
                        description="系统降级时生成的占位景点,建议稍后重试获取实时推荐",
                        category="景点",
                    )
                ],
                meals=[
                    Meal(type="breakfast", name="早餐", description="酒店或附近早餐"),
                    Meal(type="lunch", name="午餐", description="景区周边简餐"),
                    Meal(type="dinner", name="晚餐", description="本地特色餐厅"),
                ],
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"系统已返回可执行降级方案。原因: {reason}",
        )
