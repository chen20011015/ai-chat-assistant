"""
AI 对话助手 - 基于大模型API的多轮对话工具
"""

import streamlit as st
import requests
import json
import re
from datetime import datetime


def clean_safety_tags(content):
    if not content:
        return content
    for p in [r'<ds_safety>.*?</ds_safety>', r'<safety>.*?</safety>',
              r'<ds_filter>.*?</ds_filter>', r'<filter>.*?</filter>']:
        content = re.sub(p, '', content, flags=re.DOTALL)
    return content.strip() if content.strip() else "（回复被安全过滤）"


def clean_api_key(key):
    if not key:
        return key
    return ''.join(c for c in key if ord(c) < 128).strip()


def get_default_key(provider):
    try:
        s = st.secrets.get("api", {})
        if provider == "DeepSeek":
            return s.get("deepseek_key", "")
        elif provider == "智谱GLM":
            return s.get("zhipu_key", "")
        elif provider == "OpenAI":
            return s.get("openai_key", "")
    except Exception:
        pass
    return ""


st.set_page_config(page_title="AI 对话助手", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
    .footer { font-size:11px; color:#888; text-align:center; margin-top:40px; }
    .free-badge { background-color:#4caf50; color:white; padding:2px 8px; border-radius:4px; font-size:11px; }
</style>""", unsafe_allow_html=True)

PROVIDERS = {
    "智谱GLM": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "models": ["glm-4-flash", "glm-4-plus", "glm-4-air"],
        "default_model": "glm-4-flash",
        "key_prefix": "",
        "register_url": "https://open.bigmodel.cn/",
        "register_hint": "GLM-4-Flash 免费调用",
        "support_stream_options": False,
        "max_context_tokens": 128000
    },
    "DeepSeek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "key_prefix": "sk-",
        "register_url": "https://platform.deepseek.com/",
        "register_hint": "注册送500万token",
        "support_stream_options": True,
        "max_context_tokens": 64000
    },
    "OpenAI": {
        "url": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "key_prefix": "sk-",
        "register_url": "https://platform.openai.com/",
        "register_hint": "需海外手机号",
        "support_stream_options": True,
        "max_context_tokens": 128000
    }
}


def truncate_messages(messages, max_context_tokens=64000):
    if not messages:
        return messages
    sys = [m for m in messages if m["role"] == "system"]
    chat = [m for m in messages if m["role"] != "system"]
    est = sum(len(m["content"]) for m in chat) // 4
    mx = int(max_context_tokens * 0.8)
    if est <= mx:
        return sys + chat
    while chat and est > mx:
        est -= len(chat.pop(0)["content"]) // 4
    return sys + chat


def call_api(cfg, messages, key, model, temp, max_tok):
    key = clean_api_key(key)
    try:
        r = requests.post(cfg["url"], headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": model, "messages": messages, "temperature": temp,
                                "max_tokens": max_tok, "stream": False}, timeout=120)
        if r.status_code == 401:
            return None, {"error": "❌ API Key 无效"}
        elif r.status_code == 429:
            return None, {"error": "❌ 请求过于频繁"}
        elif r.status_code == 402:
            return None, {"error": "❌ 额度不足"}
        elif r.status_code != 200:
            return None, {"error": f"❌ 错误：{r.status_code}"}
        res = r.json()
        if "error" in res:
            return None, {"error": f"❌ {res['error'].get('message', '未知错误')}"}
        content = clean_safety_tags(res["choices"][0]["message"]["content"])
        u = res.get("usage", {})
        return content, {"total_tokens": u.get("total_tokens", 0), "model": model, "status": "ok"}
    except requests.exceptions.Timeout:
        return None, {"error": "❌ 请求超时"}
    except requests.exceptions.ConnectionError:
        return None, {"error": "❌ 无法连接服务器"}
    except Exception as e:
        return None, {"error": f"❌ {str(e)}"}


def stream_api(cfg, messages, key, model, temp, max_tok):
    key = clean_api_key(key)
    data = {"model": model, "messages": messages, "temperature": temp,
            "max_tokens": max_tok, "stream": True}
    if cfg.get("support_stream_options"):
        data["stream_options"] = {"include_usage": True}
    try:
        r = requests.post(cfg["url"], headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json=data, timeout=120, stream=True)
        if r.status_code == 401:
            yield "❌ API Key 无效", None
            return
        elif r.status_code != 200:
            yield f"❌ 错误：{r.status_code}", None
            return
        full = ""
        usage = None
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
                ch = chunk.get("choices", [])
                if ch and "content" in ch[0].get("delta", {}):
                    full += ch[0]["delta"]["content"]
                    yield clean_safety_tags(full), None
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        info = {"model": model, "status": "ok", "stream": True}
        if usage:
            info["total_tokens"] = usage.get("total_tokens", 0)
        yield clean_safety_tags(full), info
    except Exception as e:
        yield f"❌ {str(e)}", None


defaults = {"messages": [], "api_key": "", "provider": "智谱GLM", "model": "glm-4-flash",
            "temperature": 0.7, "max_tokens": 2000, "system_prompt": "你是一个有帮助的AI助手。",
            "debug_mode": False, "stream_mode": True, "total_tokens_used": 0}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.header("⚙️ 设置")
    st.session_state.provider = st.selectbox("选择 API 提供商", list(PROVIDERS.keys()))
    cfg = PROVIDERS[st.session_state.provider]
    st.session_state.model = st.selectbox("选择模型", cfg["models"],
        index=cfg["models"].index(cfg["default_model"]))
    default_key = get_default_key(st.session_state.provider)
    current_key = st.session_state.api_key or default_key
    st.session_state.api_key = st.text_input("API Key", value=current_key, type="password",
        help="已预配置免费Key，可直接使用")
    if default_key:
        st.markdown('<span class="free-badge">✅ 已预配置免费Key</span>', unsafe_allow_html=True)
    st.markdown(f"📋 [注册获取 Key]({cfg['register_url']})")
    st.caption(cfg["register_hint"])
    st.divider()
    st.subheader("高级参数")
    st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)
    st.session_state.max_tokens = st.slider("Max Tokens", 100, 4000, st.session_state.max_tokens, 100)
    st.session_state.system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=80)
    st.divider()
    st.subheader("功能")
    st.session_state.stream_mode = st.checkbox("⚡ 流式输出", st.session_state.stream_mode)
    st.session_state.debug_mode = st.checkbox("🔧 调试模式", st.session_state.debug_mode)
    st.divider()
    if st.session_state.total_tokens_used > 0:
        st.metric("累计消耗", f"{st.session_state.total_tokens_used:,} tokens")
    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens_used = 0
        st.rerun()
    if st.button("📥 导出对话", use_container_width=True) and st.session_state.messages:
        txt = f"# AI对话记录\n\n提供商：{st.session_state.provider} | 模型：{st.session_state.model}\n\n"
        for m in st.session_state.messages:
            txt += f"### {'👤' if m['role']=='user' else '🤖'}\n{m['content']}\n\n---\n\n"
        st.download_button("下载", txt, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", "text/markdown")

st.title("🤖 AI 对话助手")
st.caption("支持 DeepSeek / 智谱GLM / OpenAI | 打开即用")

if not st.session_state.api_key:
    st.warning("⚠️ 请在左侧填入 API Key")
    st.stop()

if not st.session_state.messages:
    st.markdown("<div style='text-align:center; padding:40px; color:#888;'><h3>👋 你好！我是AI对话助手</h3></div>",
        unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(clean_safety_tags(m["content"]))
        if "usage" in m and m["usage"].get("total_tokens"):
            st.caption(f"📊 {m['usage']['total_tokens']} tokens")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("输入消息", placeholder="在这里输入...", height=100, label_visibility="collapsed")
    submit = st.form_submit_button("发送", use_container_width=True)

if submit and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    with st.chat_message("user"):
        st.markdown(user_input.strip())
    msgs = [{"role": "system", "content": st.session_state.system_prompt}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    msgs = truncate_messages(msgs, cfg.get("max_context_tokens", 64000))

    if st.session_state.stream_mode:
        with st.chat_message("assistant"):
            ph = st.empty()
            full = ""
            ui = None
            for partial, usage in stream_api(cfg, msgs, st.session_state.api_key,
                st.session_state.model, st.session_state.temperature, st.session_state.max_tokens):
                full = partial
                ph.markdown(full + "▌")
                if usage:
                    ui = usage
            ph.markdown(full)
            if ui and ui.get("total_tokens"):
                st.caption(f"📊 {ui['total_tokens']} tokens")
                st.session_state.total_tokens_used += ui["total_tokens"]
            if st.session_state.debug_mode and ui:
                with st.expander("🔧 调试"):
                    st.json(ui)
            md = {"role": "assistant", "content": full}
            if ui:
                md["usage"] = ui
            st.session_state.messages.append(md)
    else:
        with st.spinner("AI 正在思考..."):
            resp, ui = call_api(cfg, msgs, st.session_state.api_key,
                st.session_state.model, st.session_state.temperature, st.session_state.max_tokens)
        if resp is None:
            err = ui.get("error", "未知错误") if ui else "未知错误"
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            with st.chat_message("assistant"):
                st.markdown(resp)
                if ui and ui.get("total_tokens"):
                    st.caption(f"📊 {ui['total_tokens']} tokens")
                    st.session_state.total_tokens_used += ui["total_tokens"]
                if st.session_state.debug_mode and ui:
                    with st.expander("🔧 调试"):
                        st.json(ui)
            md = {"role": "assistant", "content": resp}
            if ui:
                md["usage"] = ui
            st.session_state.messages.append(md)
    st.rerun()

st.divider()
st.markdown('<div class="footer">🔧 Python + Streamlit + 大模型API</div>', unsafe_allow_html=True)
