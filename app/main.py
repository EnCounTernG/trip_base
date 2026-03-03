"""FastAPI 服务入口。

运行后提供：
- GET /health: 健康检查
- POST /chat: 旅行助手对话接口
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.graph import run_agent

app = FastAPI(title="Trip Assistant Phase A", version="0.1.0")


class ChatRequest(BaseModel):
    """对话请求体。"""

    query: str = Field(..., description="用户输入")
    max_iterations: int = Field(4, ge=1, le=8, description="ReAct 最大迭代次数")


@app.get("/health")
def health() -> dict:
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """对话接口：调用 LangGraph Agent 返回结果。"""
    return run_agent(user_query=req.query, max_iterations=req.max_iterations)
