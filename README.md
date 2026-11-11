# AI 对话助手

基于大模型 API 的多轮对话 Web 应用，支持 **DeepSeek / 智谱 GLM / OpenAI** 一键切换。

[![在线 Demo](https://img.shields.io/badge/在线%20Demo-点击体验-1A5FB4?style=flat-square)](https://ai-chat-assistant-jpvo3l7ktkmmvhpezca3p.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**在线体验：** https://ai-chat-assistant-jpvo3l7ktkmmvhpezca3p.streamlit.app

演示环境已内置 Key，打开即可对话，无需自行申请。

---

## 功能

- 多轮对话，维护完整对话历史
- **三家大模型 API 一键切换**：DeepSeek / 智谱 GLM / OpenAI
- 流式输出，首字延迟可感知地降低
- Temperature、Max Tokens、System Prompt 运行时调节
- 对话记录一键导出 Markdown
- 响应式界面，适配移动端
- 错误分类提示：Key 无效 / 余额不足 / 网络不通，不让用户猜
- 单会话轮数上限，防止公开演示被刷额度

## 技术栈

| 组件 | 技术 |
|---|---|
| 前端框架 | Streamlit |
| 后端语言 | Python 3.10+ |
| API 调用 | requests（流式 SSE 解析） |
| 部署平台 | Streamlit Community Cloud |

## 设计说明

### 三家厂商为什么能用一套代码

DeepSeek、智谱 GLM、OpenAI 均兼容 OpenAI Chat Completions 协议，差异只在 `base_url`、`model` 和密钥。

因此把厂商差异收敛成一张配置表 `PROVIDERS`，请求、流式解析、错误处理全部共用。**新增一个厂商只需在配置里加一条，业务代码零改动。**

### 为什么侧边栏默认折叠

配置项（选厂商、调参、填 Key）是给开发者用的，不是给访客看的。首屏应该是对话界面本身。

`initial_sidebar_state = "collapsed"` 让访客第一眼看到输入框和示例问题，需要调参时再展开。

## 快速开始

```bash
git clone https://github.com/chen20011015/ai-chat-assistant.git
cd ai-chat-assistant
pip install -r requirements.txt
```

配置 Key（**不要写进代码**）：

```bash
# Windows PowerShell
$env:API_KEY = "你的key"

# macOS / Linux
export API_KEY="你的key"
```

启动：

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501

### 获取 API Key

| 提供商 | 地址 | 免费额度 |
|---|---|---|
| 智谱 GLM | https://open.bigmodel.cn/ | GLM-4-Flash 免费 |
| DeepSeek | https://platform.deepseek.com/ | 注册赠送额度 |
| OpenAI | https://platform.openai.com/ | 需付费 |

## 部署到 Streamlit Community Cloud

1. 代码推到 GitHub
2. 打开 https://share.streamlit.io/ → **New app** → 选择仓库
3. Main file path 填 `app.py`
4. **App settings → Secrets** 填入：

```toml
API_KEY = "你的key"
```

5. Deploy，约 2 分钟构建完成，获得公网链接

## 项目结构

```
ai-chat-assistant/
├── app.py                  # 主应用（界面 + API 调用 + 流式解析）
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml         # 侧栏默认折叠、上传限制、主题
```

## License

MIT


## License

MIT
