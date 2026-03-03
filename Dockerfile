# 基于官方 Python 镜像构建
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 拷贝依赖并安装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码
COPY app ./app

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
