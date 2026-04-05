# TalentTalk - AI 驱动的面试模拟系统

一个基于大语言模型的智能面试模拟平台，支持文字/语音面试、简历优化和职位信息抓取。

## 🚀 功能特性

### 1. 智能面试模拟
- **文字面试**：基于 AI 的交互式文字面试体验
- **语音面试**：支持语音输入和语音输出，模拟真实面试场景
  - 使用 Whisper 进行语音识别
  - 使用 ElevenLabs 进行语音合成

### 2. 简历优化
- AI 驱动的简历分析和改进建议
- 基于目标职位的简历内容优化
- 关键词提取和匹配度分析

### 3. 职位信息抓取
- Boss 直聘职位信息自动抓取
- 支持多页面爬取
- 职位详情自动解析

### 4. 动态面试工作流
- 基于用户简历生成个性化面试问题
- 支持多种职位类型的面试场景
- 使用 LangGraph 构建智能面试流程

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| **前端** | Streamlit, streamlit-webrtc |
| **LLM** | LangChain, LangGraph, OpenAI, 阿里云 DashScope, Google Gemini |
| **语音** | AssemblyAI, Whisper, ElevenLabs |
| **向量数据库** | Milvus, ChromaDB |
| **爬虫** | Selenium/Playwright |
| **PDF 处理** | PyPDF, FPDF |

## 📦 安装

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd TalentTalk
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量模板并填写你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API 密钥：

```env
# API Keys
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 4. 获取 API 密钥

- **DashScope (阿里云)**: [阿里云百炼](https://bailian.console.aliyun.com/)
- **ElevenLabs**: [ElevenLabs 官网](https://elevenlabs.io/)
- **AssemblyAI**: [AssemblyAI 官网](https://www.assemblyai.com/)

## 🎯 使用方法

### 启动主应用

```bash
streamlit run home.py
```

### 启动语音面试应用

```bash
streamlit run voice_interview_app_whisper.py
```

### 启动简历优化器

```bash
streamlit run resume_optimizer.py
```

## 📁 项目结构

```
TalentTalk/
├── home.py                    # 主入口 - Streamlit Web 应用
├── text_interview_app.py      # 文字面试应用
├── voice_interview_app_whisper.py  # 语音面试应用
├── resume_optimizer.py        # 简历优化器
├── auto.py                    # 自动化功能
├── boss_scraper.py            # Boss 直聘爬虫
├── src/                       # 核心业务逻辑
│   ├── workflow.py            # 面试工作流
│   └── dynamic_workflow.py    # 动态工作流
├── utils/                     # 工具函数
│   └── audio_utils.py         # 音频处理
├── data/                      # 默认数据
├── uploaded_resumes/          # 用户上传的简历
└── uploaded_questions/        # 上传的问题
```

## 🔒 安全提示

- ✅ 使用 `.env.example` 作为密钥配置模板
- ✅ 定期轮换 API 密钥

## 📄 许可证

MIT License

## 🤝 参考

https://github.com/M-Mowina/TalentTalk---AI-powered-interview-system
