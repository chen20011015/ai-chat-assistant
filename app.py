# -*- coding: utf-8 -*-
"""
AI 对话助手 —— 多模型 API 聚合

支持 DeepSeek / 智谱 GLM / OpenAI 三家，统一走 OpenAI 兼容协议：
新增厂商只需在 PROVIDERS 里加一条配置，业务代码零改动。

设计要点：
1. API Key 优先读 Streamlit Secrets，访客无需自备 Key 即可体验
2. 侧边栏默认折叠，首屏直接是对话界面（配置项是给开发者用的，不该挡住访客）
3. 流式输出，首字延迟可控
4. 错误按 Key 无效 / 余额不足 / 网络不通分类提示，不让用户猜
5. 单会话轮数上限，防止公开 Demo 被刷额度
"""

import json

import requests
import streamlit as st

# ============================ 配置 ============================

st.set_page_config(
    page_title="AI 对话助手",
    layout="centered",
    initial_sidebar_state="collapsed",  # 关键：首屏折叠侧栏
)

PROVIDERS = {
    "智谱 GLM": {
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-flash", "glm-4-air", "glm-4-plus"],
        "default": "glm-4-flash",
    },
    "DeepSeek": {
        "base": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default": "deepseek-chat",
    },
    "OpenAI": {
        "base": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "default": "gpt-4o-mini",
    },
}

DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的 AI 助手，回答简洁、准确，中文优先。"
MAX_ROUNDS = 30  # 单会话最大轮数，防刷

SAMPLES = [
    "用三句话解释 RAG 是什么",
    "RAG 和微调，什么时候该选哪个",
    "怎么降低大模型的幻觉",
]


# ============================ Key 获取 ============================

def server_key() -> str:
    """优先从 Streamlit Secrets 读取管理员配置的 Key。"""
    for k in ("API_KEY", "DEMO_API_KEY"):
        try:
            v = st.secrets.get(k, "")
            if v:
                return str(v).strip()
        except Exception:
            continue
    return ""


# ============================ 会话状态 ============================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rounds" not in st.session_state:
    st.session_state.rounds = 0


# ============================ 侧边栏（默认折叠）============================

with st.sidebar:
    st.subheader("设置")

    provider_name = st.selectbox("API 提供商", list(PROVIDERS.keys()), index=0)
    cfg = PROVIDERS[provider_name]
    model = st.selectbox(
        "模型",
        cfg["models"],
        index=cfg["models"].index(cfg["default"]),
    )

    st.divider()

    st.caption("演示模式已内置 Key，可直接对话。如需用自己账号，填在下面覆盖。")
    user_key = st.text_input("API Key（可选）", type="password", placeholder="留空则使用演示 Key")

    effective_key = user_key.strip() or server_key()

    if effective_key:
        st.success("API 已就绪")
    else:
        st.error("未配置 Key：请在 Secrets 中设置 API_KEY，或在上方填写")

    st.divider()

    st.caption("高级参数")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 4096, 2000, 128)
    system_prompt = st.text_area("System Prompt", DEFAULT_SYSTEM_PROMPT, height=90)

    st.divider()

    col_a, col_b = st.columns(2)
    if col_a.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rounds = 0
        st.rerun()

    if st.session_state.messages:
        md = "\n\n".join(
            f"**{'你' if m['role'] == 'user' else 'AI'}**：{m['content']}"
            for m in st.session_state.messages
        )
        col_b.download_button(
            "导出记录",
            data=md.encode("utf-8"),
            file_name="chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.caption(
        "技术栈：Python · Streamlit · requests\n\n"
        "[查看源码](https://github.com/chen20011015/ai-chat-assistant)"
    )


# ============================ 流式调用 ============================

class ApiError(Exception):
    """业务可读的 API 错误。"""


def stream_chat(base, key, model, messages, temperature, max_tokens):
    """统一 OpenAI 兼容协议的流式调用，逐块 yield 文本。"""
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        resp = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=(10, 120)
        )
    except requests.exceptions.ConnectTimeout:
        raise ApiError("连接超时：检查网络或代理设置")
    except requests.exceptions.ConnectionError:
        raise ApiError("网络不通：无法连接到 API 服务器")
    except requests.exceptions.RequestException as e:
        raise ApiError(f"请求失败：{e}")

    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:200]
        if resp.status_code in (401, 403):
            raise ApiError(f"Key 无效或无权限（{resp.status_code}）：{detail}")
        if resp.status_code == 429:
            raise ApiError(f"触发限流或余额不足（429）：{detail}")
        if resp.status_code >= 500:
            raise ApiError(f"服务端异常（{resp.status_code}）：{detail}")
        raise ApiError(f"请求被拒绝（{resp.status_code}）：{detail}")

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        chunk = raw[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            yield piece


# ============================ 页面主体 ============================

st.title("AI 对话助手")
st.caption("DeepSeek / 智谱 GLM / OpenAI 一键切换 · 流式输出 · 无需自备 Key")

# 历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 空态：给三个示例问题，降低冷启动门槛
if not st.session_state.messages:
    st.write("")
    st.markdown("**试试这些**")
    cols = st.columns(3)
    for col, q in zip(cols, SAMPLES):
        if col.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# 输入框
prompt = st.chat_input("在这里输入…")
if prompt:
    if st.session_state.rounds >= MAX_ROUNDS:
        st.warning(f"演示额度已用完（{MAX_ROUNDS} 轮），请刷新页面或填写自己的 API Key。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# 若最后一条是用户消息，则生成回复
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        if not effective_key:
            st.error("未配置 API Key，无法生成回复。")
        else:
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend(st.session_state.messages)

            try:
                reply = st.write_stream(
                    stream_chat(
                        cfg["base"],
                        effective_key,
                        model,
                        api_messages,
                        temperature,
                        max_tokens,
                    )
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
                st.session_state.rounds += 1
            except ApiError as e:
                st.error(str(e))
                # 出错时回滚这条用户消息，避免历史里留空回复
                st.session_state.messages.pop()
            except Exception as e:  # 兜底，避免整页崩掉
                st.error(f"未预期的错误：{type(e).__name__}: {e}")
                st.session_state.messages.pop()


st.divider()
st.markdown('<div class="footer">🔧 Python + Streamlit + 大模型API</div>', unsafe_allow_html=True)
