"""
AI 对话助手 - 基于大模型API的多轮对话工具
支持：DeepSeek / 智谱GLM / OpenAI 三种API切换
功能：多轮对话、历史记录、温度调节、流式输出
"""

import streamlit as st
import requests
import json
import time
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
    .chat-message {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        max-width: 85%;
    }
    .user-message {
        background-color: #1a237e;
        color: white;
        margin-left: 15%;
    }
    .assistant-message {
        background-color: #f0f0f0;
        color: #222;
        margin-right: 15%;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============ API 调用函数 ============

def call_deepseek_api(messages, api_key, temperature=0.7, max_tokens=2000):
    """调用 DeepSeek API"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请检查网络或重试"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接服务器，请检查网络"
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return "❌ API Key 无效，请检查密钥是否正确"
        elif response.status_code == 429:
            return "❌ 请求过于频繁，请稍后重试"
        return f"❌ 服务器返回错误：{response.status_code}"
    except Exception as e:
        return f"❌ 发生错误：{str(e)}"


def call_zhipu_api(messages, api_key, temperature=0.7, max_tokens=2000):
    """调用智谱 GLM API"""
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "glm-4-flash",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请检查网络或重试"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接服务器，请检查网络"
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return "❌ API Key 无效，请检查密钥是否正确"
        elif response.status_code == 429:
            return "❌ 请求过于频繁，请稍后重试"
        return f"❌ 服务器返回错误：{response.status_code}"
    except Exception as e:
        return f"❌ 发生错误：{str(e)}"


def call_openai_api(messages, api_key, temperature=0.7, max_tokens=2000):
    """调用 OpenAI API（兼容格式，也适用于其他 OpenAI 兼容接口）"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请检查网络或重试"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接服务器，请检查网络"
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return "❌ API Key 无效，请检查密钥是否正确"
        elif response.status_code == 429:
            return "❌ 请求过于频繁，请稍后重试"
        return f"❌ 服务器返回错误：{response.status_code}"
    except Exception as e:
        return f"❌ 发生错误：{str(e)}"


# ============ Session State 初始化 ============

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "provider" not in st.session_state:
    st.session_state.provider = "DeepSeek"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2000

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "你是一个有帮助的AI助手。"

# ============ 侧边栏设置 ============

with st.sidebar:
    st.header("⚙️ 设置")

    # API 提供商选择
    st.session_state.provider = st.selectbox(
        "选择 API 提供商",
        ["DeepSeek", "智谱GLM", "OpenAI"],
        help="选择你要使用的大模型API"
    )

    # API Key 输入
    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="在下方链接注册获取API Key"
    )
    st.session_state.api_key = api_key_input

    # 根据提供商显示注册链接
    if st.session_state.provider == "DeepSeek":
        st.markdown("📋 [注册获取 Key](https://platform.deepseek.com/)")
        st.caption("DeepSeek 注册送500万token免费额度")
    elif st.session_state.provider == "智谱GLM":
        st.markdown("📋 [注册获取 Key](https://open.bigmodel.cn/)")
        st.caption("智谱GLM-4-Flash 模型免费调用")
    elif st.session_state.provider == "OpenAI":
        st.markdown("📋 [注册获取 Key](https://platform.openai.com/)")
        st.caption("需要海外手机号注册")

    st.divider()

    # 高级参数
    st.subheader("高级参数")

    st.session_state.temperature = st.slider(
        "Temperature（创造性）",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="0=精确回答，2=高度创造性"
    )

    st.session_state.max_tokens = st.slider(
        "Max Tokens（最大长度）",
        min_value=100,
        max_value=4000,
        value=st.session_state.max_tokens,
        step=100,
        help="AI回复的最大token数"
    )

    st.session_state.system_prompt = st.text_area(
        "System Prompt（系统提示词）",
        value=st.session_state.system_prompt,
        height=80,
        help="设定AI的角色和行为"
    )

    st.divider()

    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # 导出对话
    if st.button("📥 导出对话记录", use_container_width=True):
        if st.session_state.messages:
            export_text = f"# AI对话记录 - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
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

# API Key 检查
if not st.session_state.api_key:
    st.warning("⚠️ 请在左侧侧边栏填入 API Key 才能开始对话")
    st.info("💡 **没有Key？** 左侧有注册链接，DeepSeek和智谱都有免费额度")
    st.stop()

# ============ 显示对话历史 ============

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # 欢迎消息
        st.markdown("""
        <div style='text-align:center; padding:40px; color:#888;'>
            <h3>👋 你好！我是AI对话助手</h3>
            <p>在下方输入框开始对话吧</p>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"<div class='chat-message user-message'>👤 <strong>你：</strong><br>{message['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-message assistant-message'>🤖 <strong>AI：</strong><br>{message['content']}</div>",
                unsafe_allow_html=True
            )

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
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip()
    })

    # 构建API调用消息（含系统提示）
    api_messages = [{"role": "system", "content": st.session_state.system_prompt}]
    api_messages.extend(st.session_state.messages)

    # 调用对应API
    with st.spinner("AI 正在思考..."):
        if st.session_state.provider == "DeepSeek":
            response = call_deepseek_api(
                api_messages,
                st.session_state.api_key,
                st.session_state.temperature,
                st.session_state.max_tokens
            )
        elif st.session_state.provider == "智谱GLM":
            response = call_zhipu_api(
                api_messages,
                st.session_state.api_key,
                st.session_state.temperature,
                st.session_state.max_tokens
            )
        elif st.session_state.provider == "OpenAI":
            response = call_openai_api(
                api_messages,
                st.session_state.api_key,
                st.session_state.temperature,
                st.session_state.max_tokens
            )

    # 添加AI回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()

# ============ 页脚 ============

st.divider()
st.caption("🔧 技术栈：Python + Streamlit + 大模型API | [GitHub源码]() | 部署于 Streamlit Cloud")
