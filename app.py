"""
AI 对话助手 - 基于大模型API的多轮对话工具
支持：DeepSeek / 智谱GLM / OpenAI 三种API切换
功能：多轮对话、历史记录、温度调节、对话导出、流式输出、Token用量显示
"""

import streamlit as st
import requests
import json
import html
from datetime import datetime

# ============ 页面配置 ============
st.set_page_config(
    page_title="AI 对话助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义样式 ============
st.markdown("""
<style>
    .footer { font-size: 11px; color: #888; text-align: center; margin-top: 40px; }
    .debug-box { background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
                 padding: 10px; font-family: monospace; font-size: 11px; margin: 8px 0;
                 word-break: break-all; white-space: pre-wrap; }
    .token-info { font-size: 11px; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ============ 模型配置 ============

PROVIDERS = {
    "DeepSeek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "key_prefix": "sk-",
        "register_url": "https://platform.deepseek.com/",
        "register_hint": "DeepSeek 注册送500万token免费额度"
    },
    "智谱GLM": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "models": ["glm-4-flash", "glm-4-plus", "glm-4-air"],
        "default_model": "glm-4-flash",
        "key_prefix": "",
        "register_url": "https://open.bigmodel.cn/",
        "register_hint": "智谱GLM-4-Flash 模型免费调用"
    },
    "OpenAI": {
        "url": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "key_prefix": "sk-",
        "register_url": "https://platform.openai.com/",
        "register_hint": "需要海外手机号注册"
    }
}

# ============ API 调用函数 ============

def call_api(provider_config, messages, api_key, model, temperature, max_tokens, stream=False):
    """统一的API调用函数，返回 (content, usage_info) 或流式响应对象"""
    url = provider_config["url"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=120, stream=stream)

        if response.status_code == 401:
            return None, {"error": "❌ API Key 无效（401），请检查密钥是否正确"}
        elif response.status_code == 429:
            return None, {"error": "❌ 请求过于频繁（429），请稍后重试"}
        elif response.status_code == 402:
            return None, {"error": "❌ API额度不足（402），请充值或更换Key"}
        elif response.status_code != 200:
            return None, {"error": f"❌ 服务器返回错误：{response.status_code}，响应：{response.text[:200]}"}

        if stream:
            return response, None
        else:
            result = response.json()

            if "error" in result:
                err = result["error"]
                return None, {"error": f"❌ API错误：{err.get('message', '未知错误')}"}

            content = result["choices"][0]["message"]["content"]

            usage = result.get("usage", {})
            usage_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "model": model,
                "status": "ok"
            }

            if not content or content.strip() == "":
                return "（AI返回了空内容，请重试或调整参数）", usage_info

            return content, usage_info

    except requests.exceptions.Timeout:
        return None, {"error": "❌ 请求超时（120秒），请检查网络或稍后重试"}
    except requests.exceptions.ConnectionError:
        return None, {"error": "❌ 无法连接服务器，请检查网络"}
    except json.JSONDecodeError:
        return None, {"error": f"❌ 返回格式错误，原始响应：{response.text[:300]}"}
    except Exception as e:
        return None, {"error": f"❌ 发生错误：{str(e)}"}


def stream_api(provider_config, messages, api_key, model, temperature, max_tokens):
    """流式调用API，逐字返回"""
    url = provider_config["url"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=120, stream=True)

        if response.status_code == 401:
            yield "❌ API Key 无效（401），请检查密钥是否正确", None
            return
        elif response.status_code == 429:
            yield "❌ 请求过于频繁（429），请稍后重试", None
            return
        elif response.status_code != 200:
            yield f"❌ 服务器返回错误：{response.status_code}", None
            return

        full_content = ""
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    full_content += delta["content"]
                    yield full_content, None
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

        yield full_content, {"model": model, "status": "ok", "stream": True}

    except requests.exceptions.Timeout:
        yield "❌ 请求超时，请检查网络或稍后重试", None
    except requests.exceptions.ConnectionError:
        yield "❌ 无法连接服务器，请检查网络", None
    except Exception as e:
        yield f"❌ 发生错误：{str(e)}", None


# ============ Session State 初始化 ============

defaults = {
    "messages": [],
    "api_key": "",
    "provider": "DeepSeek",
    "model": "deepseek-chat",
    "temperature": 0.7,
    "max_tokens": 2000,
    "system_prompt": "你是一个有帮助的AI助手。",
    "debug_mode": False,
    "stream_mode": True,
    "total_tokens_used": 0
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============ 侧边栏设置 ============

with st.sidebar:
    st.header("⚙️ 设置")

    st.session_state.provider = st.selectbox(
        "选择 API 提供商",
        list(PROVIDERS.keys()),
        help="选择你要使用的大模型API"
    )

    provider_cfg = PROVIDERS[st.session_state.provider]

    st.session_state.model = st.selectbox(
        "选择模型",
        provider_cfg["models"],
        index=provider_cfg["models"].index(provider_cfg["default_model"]),
        help="不同模型能力和价格不同"
    )

    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="在下方链接注册获取API Key"
    )
    st.session_state.api_key = api_key_input

    if st.session_state.api_key and provider_cfg["key_prefix"]:
        if not st.session_state.api_key.startswith(provider_cfg["key_prefix"]):
            st.warning(f"⚠️ 该API的Key通常以 `{provider_cfg['key_prefix']}` 开头，请检查")

    st.markdown(f"📋 [注册获取 Key]({provider_cfg['register_url']})")
    st.caption(provider_cfg["register_hint"])

    st.divider()

    st.subheader("高级参数")

    st.session_state.temperature = st.slider(
        "Temperature（创造性）",
        min_value=0.0, max_value=2.0,
        value=st.session_state.temperature, step=0.1,
        help="0=精确回答，2=高度创造性"
    )

    st.session_state.max_tokens = st.slider(
        "Max Tokens（最大长度）",
        min_value=100, max_value=4000,
        value=st.session_state.max_tokens, step=100,
        help="AI回复的最大token数"
    )

    st.session_state.system_prompt = st.text_area(
        "System Prompt（系统提示词）",
        value=st.session_state.system_prompt,
        height=80,
        help="设定AI的角色和行为"
    )

    st.divider()

    st.subheader("功能")

    st.session_state.stream_mode = st.checkbox(
        "⚡ 流式输出（逐字显示）",
        value=st.session_state.stream_mode,
        help="开启后AI回复逐字显示，体验更好"
    )

    st.session_state.debug_mode = st.checkbox(
        "🔧 调试模式",
        value=st.session_state.debug_mode,
        help="显示API调用的详细信息"
    )

    st.divider()

    if st.session_state.total_tokens_used > 0:
        st.subheader("📊 Token用量")
        st.metric("本次会话累计消耗", f"{st.session_state.total_tokens_used:,} tokens")

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens_used = 0
        st.rerun()

    if st.button("📥 导出对话记录", use_container_width=True):
        if st.session_state.messages:
            export_text = f"# AI对话记录 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            export_text += f"提供商：{st.session_state.provider} | 模型：{st.session_state.model}\n\n"
            for msg in st.session_state.messages:
                role_label = "👤 用户" if msg["role"] == "user" else "🤖 AI"
                export_text += f"### {role_label}\n{msg['content']}\n\n---\n\n"
            st.download_button(
                label="下载 Markdown 文件",
                data=export_text,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("暂无对话记录可导出")

# ============ 主界面 ============

st.title("🤖 AI 对话助手")
st.caption("基于大模型API的多轮对话工具 | 支持 DeepSeek / 智谱GLM / OpenAI")

if not st.session_state.api_key:
    st.warning("⚠️ 请在左侧侧边栏填入 API Key 才能开始对话")
    st.info("💡 **没有Key？** 左侧有注册链接，DeepSeek和智谱都有免费额度")
    st.stop()

# ============ 显示对话历史 ============

if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding:40px; color:#888;'>
        <h3>👋 你好！我是AI对话助手</h3>
        <p>在下方输入框开始对话吧</p>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if "usage" in message:
                u = message["usage"]
                if u.get("total_tokens"):
                    st.caption(f"📊 本次消耗：{u['total_tokens']} tokens")

# ============ 输入框 ============

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "输入消息",
        placeholder="在这里输入你的问题...",
        height=100,
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button("发送", use_container_width=True)

# ============ 处理发送 ============

if submit_button and user_input.strip():
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip()
    })

    with st.chat_message("user"):
        st.markdown(user_input.strip())

    api_messages = [{"role": "system", "content": st.session_state.system_prompt}]
    api_messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

    provider_cfg = PROVIDERS[st.session_state.provider]

    if st.session_state.stream_mode:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            usage_info = None

            for partial, usage in stream_api(
                provider_cfg, api_messages,
                st.session_state.api_key,
                st.session_state.model,
                st.session_state.temperature,
                st.session_state.max_tokens
            ):
                full_response = partial
                placeholder.markdown(full_response + "▌")
                if usage:
                    usage_info = usage

            placeholder.markdown(full_response)

            if usage_info and usage_info.get("total_tokens"):
                st.caption(f"📊 本次消耗：{usage_info['total_tokens']} tokens")
                st.session_state.total_tokens_used += usage_info["total_tokens"]

            if st.session_state.debug_mode and usage_info:
                with st.expander("🔧 调试信息"):
                    st.json(usage_info)

            msg_data = {"role": "assistant", "content": full_response}
            if usage_info:
                msg_data["usage"] = usage_info
            st.session_state.messages.append(msg_data)

    else:
        with st.spinner("AI 正在思考..."):
            response, usage_info = call_api(
                provider_cfg, api_messages,
                st.session_state.api_key,
                st.session_state.model,
                st.session_state.temperature,
                st.session_state.max_tokens,
                stream=False
            )

        if response is None:
            error_msg = usage_info.get("error", "未知错误") if usage_info else "未知错误"
            with st.chat_message("assistant"):
                st.error(error_msg)
                if st.session_state.debug_mode and usage_info:
                    with st.expander("🔧 调试信息"):
                        st.json(usage_info)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.chat_message("assistant"):
                st.markdown(response)
                if usage_info and usage_info.get("total_tokens"):
                    st.caption(f"📊 本次消耗：{usage_info['total_tokens']} tokens")
                    st.session_state.total_tokens_used += usage_info["total_tokens"]

                if st.session_state.debug_mode and usage_info:
                    with st.expander("🔧 调试信息"):
                        st.json(usage_info)

            msg_data = {"role": "assistant", "content": response}
            if usage_info:
                msg_data["usage"] = usage_info
            st.session_state.messages.append(msg_data)

    st.rerun()

# ============ 页脚 ============

st.divider()
st.markdown(
    '<div class="footer">🔧 技术栈：Python + Streamlit + 大模型API | 部署于 Streamlit Cloud</div>',
    unsafe_allow_html=True
)
