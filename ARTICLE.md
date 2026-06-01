# agent-llm-mock：让 AI Agent 调试不再依赖真实 LLM

## 背景

2024-2026 年，AI Agent 开发进入了爆发期。从 Claude Code 到 Cursor，从 LangChain 到 CrewAI，几乎每个 AI 工具都在让 LLM 调用外部工具、操作文件、执行命令。

但开发这些 Agent 功能时，一个尴尬的问题始终存在：**你怎么调试工具调用逻辑？**

真实 LLM 有三个致命问题：

1. **不可预测** — 同一个 prompt，两次调用可能返回不同的 tool_call，你永远不知道下一次会调用哪个工具、传什么参数
2. **烧钱** — 调试一个多轮 tool-calling 对话，动辄几十上百次 API 调用。GPT-4 的 tool_calls 每轮可能消耗数万 tokens，调试一天轻松花掉几十美元
3. **慢** — 每次调试都要等 API 响应，网络延迟 + 模型推理时间，迭代效率极低

更麻烦的是**边界场景**：

- 你的 Agent 代码能正确处理 tool_calls 为空的情况吗？
- 工具调用失败时的 fallback 逻辑对吗？
- 多个 tool_calls 同时返回时的执行顺序正确吗？
- Anthropic 和 OpenAI 两种 tool 格式的互转有没有 bug？

这些场景靠真实 LLM 几乎无法稳定复现。

## agent-llm-mock 是什么

**一句话：一个本地 LLM Mock 服务器，让你在无网络、零成本、完全可控的条件下调试 AI Agent 的工具调用行为。**

它同时兼容 OpenAI 和 Anthropic 两种 API 格式，提供了一个 Web 控制台，让你可以：

- 看到 Agent 发来的完整请求（messages、tools、parameters）
- **手动决定**返回什么内容、调用哪些工具、传什么参数
- 或者把请求**转发到真实 LLM**，同时检查上游返回了什么
- 也可以用**预设脚本**自动匹配和响应

本质上，它是架在你 Agent 和 LLM 之间的**透明代理 + 手动控制台**。

## 解决的痛点

### 痛点 1：调试工具调用靠"猜"

你用 Claude Code 写了个 Agent，它调了 5 个工具，你只能看日志里的 JSON，看不清工具的参数 schema、分不清哪个 tool_call 对应哪个工具定义。

**解决：** Dashboard 把每个工具的参数表单化展示——string 是输入框，boolean 是下拉框，enum 是选择器，object/array 是 textarea。工具的 name、description、必填标记一目了然。

### 痛点 2：不敢测异常路径

你想测"LLM 不返回 tool_call 只返回文本"的情况，但没法让LLM稳定地不调工具。

**解决：** 手动模式下，你完全控制返回内容——可以只填文本不勾选任何工具，也可以勾选 3 个工具填不同的参数。每个边界场景 1 秒内构造完毕。

### 痛点 3：转发模式下看不见工具调用

你用 DeepSeek 的 API 跑 Agent，返回的 tool_calls 嵌在一大坨 JSON 里，肉眼解析费劲。

**解决：** Forward 模式自动从上游响应中提取 tool_calls，交叉匹配请求中的工具定义，展示每个工具的 name、arguments、description。OpenAI 和 Anthropic 两种格式都支持，流式和非流式都支持。

### 痛点 4：多轮对话调试成本高

Agent 跑一轮对话要调 5 次 API，每次 $0.01，一天调试 100 轮就是 $5。一个月上千块花在调试上。

**解决：** agent-llm-mock 完全本地运行，零 API 费用。只有在需要用真实 LLM 验证时才 forward，其他时候全部手动 mock。

## 三种工作模式

### 模式 1：脚本模式（Scripted）

适合**确定性测试用例**。写一个 JSON 规则文件，当请求内容匹配关键字时自动返回预设响应：

```json
[
  {
    "match": {"text_contains": ["查天气", "get_weather"]},
    "response": {
      "content": "好的，我来帮你查天气",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\": \"Beijing\"}"
          }
        }
      ]
    }
  }
]
```

启动时加载：`agent-llm-mock --scripts rules.json`

你的 CI 测试里 Agent 发出的请求会被自动匹配并返回预设响应，无需人工干预。

### 模式 2：转发模式（Forward）

适合**需要真实 LLM 但想观察 tool_calls**的场景。配置目标 API：

```json
[
  {
    "match": {"api_format": "openai"},
    "target_url": "https://api.deepseek.com/v1",
    "timeout": 30
  }
]
```

启动：`agent-llm-mock --forward-config fw.json`

请求被透明代理到真实 API，Dashboard 里能看到：
- 完整的 upstream request/response
- 自动提取的 tool_calls（带 description）
- 请求中定义的所有工具列表（参数类型、是否必填）

### 模式 3：手动模式（Manual）

默认模式，**最灵活**。所有请求排队显示在 Dashboard，你手动决定返回什么：

- 输入文本响应
- 勾选要调用的工具
- 填写每个工具的参数
- 点击 Submit

适合：
- 开发新功能时快速验证逻辑
- 复现 bug 时精确控制 LLM 输出
- Demo 演示时不需要依赖外部 API

## 应用场景

### 场景 1：Agent 开发者日常调试

```
你的 Agent 代码 → agent-llm-mock (手动模式) → 你决定返回什么
```

开发时把 `base_url` 指向 `localhost:9999/v1`，每当你 Agent 发出请求，Dashboard 弹出通知。你手动构造响应，验证 Agent 的下一步行为是否符合预期。

### 场景 2：CI 自动化测试

```
pytest → Agent 代码 → agent-llm-mock (脚本模式) → 预设响应
```

在 CI 中启动 agent-llm-mock + 脚本配置，Agent 的每次 API 调用都被规则文件精确匹配。测试结果是确定性的——同样的输入永远返回同样的输出。

```bash
# CI 测试脚本示例
agent-llm-mock --port 9999 --scripts test_rules.json &
sleep 2
pytest tests/
```

### 场景 3：工具定义调试

你定义了 20 个 tools 给 Agent，需要验证：
- 参数 schema 是否正确
- description 是否清晰
- required 字段是否合理

把 Agent 指向 agent-llm-mock，在 Dashboard 里直接看到所有工具的完整定义——name、description、parameters（展开为表单字段）。一目了然地发现问题。

### 场景 4：Prompt 调试

改了 system prompt 后，不确定 Agent 会如何理解 tools 的用法。用 Forward 模式代理到真实 LLM，在 Dashboard 里同时看到：
- 你发送的 prompt + tools
- LLM 返回的 content + tool_calls
- 工具调用的参数是否合理

### 场景 5：多格式兼容性测试

你的 Agent 需要同时支持 OpenAI 和 Anthropic 两种 API 格式。agent-llm-mock 同时暴露两个端点：

- `POST /v1/chat/completions`（OpenAI）
- `POST /v1/messages`（Anthropic）

两种格式的 tools 在 Dashboard 里统一展示，切换 API 格式只需要改 `base_url`。

## 架构

```
                         POST /v1/chat/completions
  AI Agent ───────────── POST /v1/messages ─────────────┐
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  agent-llm-mock   │
                                               │                   │
                          ┌────────────────────┤ 1. Script match? │
                          │                    │ 2. Forward match? │
                          │                    │ 3. Queue (manual) │
                          │                    └────────┬─────────┘
                          │                             │
              ┌───────────┴───────────┐        ┌────────┴────────┐
              │  Script Rule Match     │        │  Forward Proxy   │
              │  → Return preset JSON  │        │  → Upstream LLM  │
              └───────────────────────┘        └─────────────────┘
                          │                             │
                          └──────────┬──────────────────┘
                                     │ WebSocket
                                     ▼
                          ┌─────────────────────┐
                          │   Web Dashboard      │
                          │   localhost:9999      │
                          │                      │
                          │  • Request inspector  │
                          │  • Response editor    │
                          │  • Tool call mocker   │
                          │  • Forward inspector  │
                          └─────────────────────┘
```

核心设计原则：

- **单文件服务器** — `server.py` 一个文件包含全部逻辑，方便 hack
- **无外部状态** — 不需要数据库，重启即清空
- **WebSocket 实时推送** — Dashboard 数据实时更新，无需轮询
- **透明代理** — Forward 模式下请求/响应完全保留，不做任何修改

## 扩展点

### 1. 添加新的 API 格式

目前支持 OpenAI 和 Anthropic。要添加新格式（如 Google Gemini、Cohere），只需：

1. 在 `PendingRequest` 中新增 `api_format` 值
2. 添加对应的请求解析逻辑（提取 model、messages、tools）
3. 添加对应的响应构建函数（`_build_xxx_response`）
4. 在 `_extract_tool_calls_from_response` 中添加新格式的提取逻辑

```python
# 示例：添加 Gemini 支持
if api_format == "gemini":
    # Gemini 的 tools 在 tools[].functionDeclarations 里
    ...
```

### 2. 持久化请求历史

当前请求只在内存中，重启即丢失。可以接入 SQLite：

```python
# 在 PendingRequest 创建后写入数据库
db.insert(request_record)
# 在 Dashboard 增加历史查询 API
@app.get("/api/history")
```

### 3. 自定义匹配规则

`_match_script` 目前支持 `text_contains` 和 `api_format`。可以扩展：

- `regex` — 正则匹配
- `header_contains` — 请求头匹配
- `model_name` — 按模型名匹配
- `tool_count` — 按工具数量范围匹配
- 组合条件（AND/OR）

### 4. 批量响应序列

模拟多轮对话：预先定义一组有序响应，Agent 每发一次请求就返回下一个响应。适合测试多步 tool-calling 流程。

```json
{
  "sequence": [
    {"content": "先查天气", "tool_calls": [...]},
    {"content": "再查路线", "tool_calls": [...]},
    {"content": "最终回答", "tool_calls": []}
  ]
}
```

### 5. Response 延迟模拟

添加可配置的响应延迟，模拟真实 LLM 的推理时间：

```python
# 在规则中添加 delay 字段
{"match": {...}, "response": {...}, "delay": 2.5}  # 延迟 2.5 秒
```

### 6. Dashboard 增强

- 请求对比视图（并排比较两次请求的 prompt/tools 差异）
- 响应模板（保存常用的响应配置，一键应用）
- 导出/导入请求为 cURL 或 Python 代码
- Token 计数显示

### 7. gRPC / MCP 协议支持

当前是 REST API。可以为 MCP (Model Context Protocol) 添加端点，让 agent-llm-mock 作为 MCP Server 运行，直接 mock 工具调用而无需经过 LLM API。

## 总结

agent-llm-mock 解决的是 AI Agent 开发中最基础但最容易被忽视的问题：**如何在可控、可复现、零成本的条件下验证工具调用逻辑**。

它不需要你改变任何代码——把 `base_url` 从 `api.openai.com` 改成 `localhost:9999`，你的 Agent 就能进入完全可控的调试模式。

开源地址：[https://github.com/your-org/agent-llm-mock](https://github.com/your-org/agent-llm-mock)
