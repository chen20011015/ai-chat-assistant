# AI 对话助手

基于大模型 API 的多轮对话 Web 应用，支持 DeepSeek / 智谱GLM / OpenAI 三种 API 切换。

## 功能

- 多轮对话，维护完整对话历史
- 三种大模型 API 一键切换（DeepSeek / 智谱GLM / OpenAI）
- Temperature、Max Tokens、System Prompt 可调
- 对话记录导出为 Markdown 文件
- 响应式 UI，支持移动端

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端框架 | Streamlit |
| 后端语言 | Python 3.10+ |
| API 调用 | requests |
| 部署平台 | Streamlit Cloud（免费） |

## 快速开始

### 1. 本地运行

```bash
# 克隆仓库
git clone https://github.com/你的用户名/ai-chat-assistant.git
cd ai-chat-assistant

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

### 2. 获取 API Key

| 提供商 | 注册地址 | 免费额度 |
|--------|----------|----------|
| DeepSeek | https://platform.deepseek.com/ | 注册送500万token |
| 智谱GLM | https://open.bigmodel.cn/ | GLM-4-Flash 模型免费 |
| OpenAI | https://platform.openai.com/ | 需付费 |

### 3. 云端部署（Streamlit Cloud）

1. 把代码推到 GitHub
2. 打开 https://share.streamlit.io/
3. 点击 "New app"，选择你的 GitHub 仓库
4. Main file path 填 `app.py`
5. 点击 Deploy，等2分钟部署完成
6. 获得在线 Demo 链接，可分享给任何人

## 项目结构

```
ai-chat-assistant/
├── app.py              # 主应用（Streamlit前端 + API调用逻辑）
├── requirements.txt    # Python依赖
├── README.md           # 项目说明
└── .streamlit/
    └── config.toml     # Streamlit配置（可选）
```

## 核心代码说明

### API 调用逻辑

每个 API 提供商封装为独立函数，统一输入输出格式：

```python
def call_deepseek_api(messages, api_key, temperature, max_tokens):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": "deepseek-chat", "messages": messages, ...}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]
```

### 多轮对话实现

使用 Streamlit 的 `session_state` 维护对话历史，每次调用 API 时把完整历史传给大模型：

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# 用户发送消息时
st.session_state.messages.append({"role": "user", "content": user_input})

# 构建API调用消息（含系统提示）
api_messages = [{"role": "system", "content": system_prompt}]
api_messages.extend(st.session_state.messages)
```

## 演示截图

（部署后在此添加截图）

## 在线 Demo

（部署后在此添加链接）

## License

MIT
