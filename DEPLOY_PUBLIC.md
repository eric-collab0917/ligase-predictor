# app.py 公开网站部署指南

本文针对当前项目（Streamlit 应用，入口 `app.py`）提供两种方案：

- 方案 A：`Streamlit Community Cloud`（免费、最省事，推荐）
- 方案 B：`Docker + 云服务器`（长期稳定、可绑定自定义域名）

---

## 0) 本地先确认能跑

```bash
cd "/Users/xu/Desktop/course project"
streamlit run app.py
```

如果本地无报错，再进行公网部署。

---

## 1) 方案 A：Streamlit Community Cloud（推荐）

### 步骤 1：把代码推到 GitHub

确保仓库里至少包含：

- `app.py`
- `app/app.py`
- `requirements.txt`（本项目已提供）

### 步骤 2：在 Streamlit Cloud 创建应用

1. 打开 [https://share.streamlit.io](https://share.streamlit.io)
2. 连接你的 GitHub 账号
3. 点击 `Create app`
4. 选择你的仓库和分支
5. `Main file path` 填：`app.py`
6. 点击 `Deploy`

部署成功后会得到一个 `https://xxx.streamlit.app` 的公开地址。

### 步骤 3：云端运行注意事项

- 这个项目会加载 `torch/transformers`，首次启动会比较慢。
- 如果应用依赖本地模型权重文件，请确保这些文件也在仓库或可在启动时下载。
- 如遇 Python 版本兼容问题，在 Advanced settings 中优先选 `Python 3.10`。

---

## 2) 方案 B：Docker + 云服务器（长期稳定）

项目已提供 `Dockerfile`，可直接容器化。

### 步骤 1：本地构建并验证镜像

```bash
cd "/Users/xu/Desktop/course project"
docker build -t enzyme-ai-app:latest .
docker run --rm -p 8501:8501 enzyme-ai-app:latest
```

浏览器访问 `http://localhost:8501` 验证是否正常。

### 步骤 2：部署到云主机

可选平台：AWS EC2、阿里云 ECS、腾讯云 CVM、DigitalOcean、Railway、Render。

把镜像或代码部署到服务器后，保证容器映射 `8501` 端口。

### 步骤 3：配置 HTTPS 和域名（推荐）

- 反向代理：Nginx 或 Caddy
- 域名示例：`app.yourdomain.com`
- 证书：Let's Encrypt（自动续期）

这样可以通过 `https://app.yourdomain.com` 对外公开访问。

---

## 3) 临时分享（5 分钟）

仅用于演示，不建议长期使用：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
cloudflared tunnel --url http://localhost:8501
```

命令输出会给一个 `trycloudflare.com` 临时公网地址。

