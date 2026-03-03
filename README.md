# trip_base

一个基于 **LangChain + LangGraph + ReAct（阶段A）** 的可运行旅行助手示例。

> 本版本目标：先把阶段 A 跑通（可部署、可调用、可观测），后续再叠加 RAG / Skills / Memory / MCP。

## 1. 当前已完成功能（阶段A）

### 1.1 LangGraph 流程编排
- 使用显式状态图编排：
  - `Intent Router`（意图识别）
  - `Planner`（ReAct 思考与动作决策）
  - `Tool Executor`（工具执行）
  - `Answer Synthesizer`（答案汇总）
- 收益：流程可读、可追踪、可扩展。

### 1.2 ReAct 最小闭环
- 每轮执行：`Thought -> Action -> Observation -> Answer`。
- 已支持动作：
  - `weather_tool`（天气查询，mock）
  - `budget_tool`（预算估算）
- 收益：复杂问题可分步处理，而不是一次性“拍脑袋”。

### 1.3 可部署 API 服务
- `GET /health`：健康检查
- `POST /chat`：对话接口，返回
  - `answer`
  - `intent`
  - `thoughts`
  - `tool_calls`
  - `observations`
  - `error`

---

## 2. 项目结构与功能对应代码

```text
app/
  main.py         # FastAPI 入口与 API 定义
  state.py        # LangGraph 的统一状态 AgentState
  tools.py        # LangChain 工具定义（天气、预算）
  graph.py        # LangGraph 节点/边与 ReAct 主流程
requirements.txt  # Python 依赖
Dockerfile      # 容器部署文件
```

### 代码对应说明
- `app/state.py`
  - 定义 `AgentState`，是全流程共享的状态容器。
- `app/tools.py`
  - 使用 `@tool` 声明 LangChain 工具。
- `app/graph.py`
  - 用 `StateGraph` 组织节点。
  - 用 `ChatPromptTemplate | RunnableLambda` 实现规划器链。
  - `run_agent()` 是主调用入口。
- `app/main.py`
  - 提供 HTTP 服务，便于本地和容器部署。

---

## 3. 对应的 Lang 系列知识点（学习导图）

### 3.1 LangChain 知识点
1. **Prompt 模板（ChatPromptTemplate）**
   - 作用：规范输入上下文，约束输出结构。
2. **Runnable 组合（`|` 管道）**
   - 作用：把 prompt、模型（本示例为规则 runnable）、后处理串成链。
3. **Tool 抽象（`@tool`）**
   - 作用：把业务能力包装成统一可调用单元。

### 3.2 LangGraph 知识点
1. **StateGraph 状态图**
   - 显式定义节点与边，替代隐式链式调用。
2. **条件边（conditional edges）**
   - 根据 planner 结果动态决定执行路径。
3. **共享状态 AgentState**
   - 在多节点间传递意图、动作、观察与最终答案。

### 3.3 ReAct 知识点
1. **Thought / Action / Observation 三段式**
2. **最大迭代控制（max_iterations）防止死循环**
3. **可观测轨迹输出（thoughts/tool_calls/observations）**

---

## 4. 运行部署所需环境

## 4.1 本地运行
- Python 3.11+
- pip

安装依赖：
```bash
pip install -r requirements.txt
```

启动服务：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 4.2 Docker 部署
构建镜像：
```bash
docker build -t trip-base-phase-a .
```

运行容器：
```bash
docker run --rm -p 8000:8000 trip-base-phase-a
```

---

## 5. 运行流程（从请求到响应）

1. 用户请求进入 `POST /chat`
2. `run_agent()` 初始化 `AgentState`
3. `intent_router_node` 识别意图
4. `planner_node` 生成 ReAct 动作
5. 如需工具，则 `tool_executor_node` 调用工具并记录 observation
6. `answer_synthesizer_node` 汇总输出答案
7. 返回结构化响应（包含可观测轨迹）

---

## 6. 接口示例

请求：
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"帮我做一个3天2人上海旅行预算，并看看天气"}'
```

返回示例（简化）：
```json
{
  "answer": "已完成工具查询：上海：多云，18~24℃，适合户外+室内混合行程。如果你愿意，我可以继续补全详细日程。",
  "intent": "plan",
  "thoughts": ["用户关注天气，先查天气再给建议。"],
  "tool_calls": [{"tool": "weather_tool", "input": {"city": "上海"}}],
  "observations": [{"tool": "weather_tool", "result": "上海：多云，18~24℃，适合户外+室内混合行程"}],
  "error": ""
}
```

---

## 7. 下一步（阶段B预告）
- 在 `app/rag/` 增加索引与检索
- 在 `app/skills/` 增加 skill router 与 skill contract
- 将 planner 从规则 runnable 升级为真实 LLM + 结构化输出
