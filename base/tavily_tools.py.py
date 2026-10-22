from typing import Literal

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessageChunk
from tavily import TavilyClient
from dotenv import load_dotenv,find_dotenv
import asyncio
import os

# 加载 .env文件
load_dotenv(find_dotenv())

# 创建tavily_client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 优化2：搜索结果缓存，同一组搜索条件不重复消耗 Tavily 额度
_search_cache: dict[tuple, dict] = {}

# 定义搜索工具
@tool
def internet_search(
        query:str,
        max_results:int =10,
        topic:Literal["general","news","finance"] = "general",
        include_raw_content:bool = False):
    """
    互联网搜索工具！
    :param query: 搜索关键字
    :param max_results: 返回结果数量
    :param topic: 主题类型
    :param include_raw_content: False精简 True 返回详细结果
    :return: 搜索结果列表
    """
    cache_key = (query, max_results, topic, include_raw_content)
    if cache_key in _search_cache:
        print(f"[search] 命中缓存，直接复用上次结果：{query}")
        return _search_cache[cache_key]
    print(f"进行网络搜索！搜索条件：{query},搜索主题类别:{topic},搜索最大的条数：{max_results}")
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content
    )
    _search_cache[cache_key] = response
    return response
# 极简初始化（自动读取OPENAI环境变量）
model = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

main_agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt="""
          你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
          你有权使用 internet_search 联网搜索工具来收集信息。
        """
)


# ======================== 重点说明==============
"""
  内容1: 执行的传入参数问题
      deepagent智能体 -> { messages:[{消息内容 HumenMessage role:"user",content:"内容"}] } 
  内容2: 执行和结果问题 
      执行方法 == langchain == 执行方法: invoke / ainvoke / stream / astream
      区别: langchain -> stream -> for chunk in stream :  chunk 到底是什么???  模型返回的增量字符 [ 123  456  789 ]
           deepagents-> stream -> for chunk  in stream : chunk 到底是什么???  每次主智能体 调用模型或者调用工具的完整返回! 
           stream -> chunk ->  model 调用哪个工具(主智能体调用了一次模型) -> tools 主智能体调用了一次模型  -> tools 主智能体又
           调用一次模型 -> model -> 主智能体调用了模型做最终结果的润色!!
           理解:
           langchain - stream -> 微观程度的stream -> 模型的增量返回  例如: rag项目 返回的字符串 delta ...
           deepagents - stream -> 宏观程度的stream -> 调用的节点流程和数量  例如: 调用一个模型本身stream,deepagents只是一次!
      选择invoke还是stream [deepagents]
          前端是否实时显示执行过程!!! 想: stream  不想: invoke
          but: 你管stream还是invoke结果都能体现调用的过程(工具/模型..)
     执行结果:
          invoke => {messages : [{model:{role:"system",content="",tool_calls:{调用工具/子智能体}},
          {"tools":{role:"tool",content:"调用工具"}},{"tools":{role:"tool",content:"调用工具"}},{model:[role:"system",content="xxxxxx最终结果",]}  
          stream => {model:{role:"system",content="",tool_calls:{调用工具/子智能体}}
                     {"tools":{role:"tool",content:"调用工具"}}
          invoke获取数据的语法: result["messages"][-1].content  理解: {messages:[HumenMessage,AIMessage,ToolMessage,AIMessage(content="")]}
          stream使用目标: 过程体现 / 主智能体调用流程 / 每个节点干什么? (调用了工具 调用子智能体,调用模型...)
              for chunk in stream:
                  chunk -> 本次到底干了啥 -> 前端sse
"""

# 运行代理
prompt = input("输入你关心的问题！")
#
# result = main_agent.invoke({
#     "messages":[{
#         "role":"user",
#         "content":f"{prompt}"
#     }]
# })
#
#
# print(f"最终结果:{result["messages"][-1].content}")

payload = {
    "messages":[
        {
            "role":"user",
            "content":f"{prompt}"
        }
    ],
}


# ======================== 过程渲染：astream_events 版 ========================
# 优点：事件已经带了语义（on_tool_start / on_tool_end / on_chat_model_stream），
#      不用再靠 node_name == "model"/"tools" 硬编码判断，节点改名也不影响。
def _text_of(content):
    """兼容 str 和 qwen 的多段 content 结构"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
        )
    return ""


async def run():
    async for event in main_agent.astream_events(payload, version="v2"):
        kind = event["event"]
        meta = event.get("metadata", {})
        node = meta.get("langgraph_node")
        # 子智能体内部的事件带 "|" 层级，过滤掉，只看主图
        if "|" in meta.get("langgraph_checkpoint_ns", ""):
            continue

        if kind == "on_chat_model_stream" and node == "model":
            # 模型 token 增量，直接打字机输出
            text = _text_of(event["data"]["chunk"].content)
            if text:
                print(text, end="", flush=True)

        elif kind == "on_tool_start" and node == "tools":
            # 主智能体决定调用工具 / 子智能体
            print()
            name = event["name"]
            args = event["data"].get("input") or {}
            if name == "task":
                print(f"[模型]决定调用子智能体:{args.get('subagent_type')},描述:{args.get('description')}")
            else:
                print(f"[模型]决定调用工具:{name},参数:{args}")

        elif kind == "on_tool_end" and node == "tools":
            # 工具 / 子智能体返回
            name = event["name"]
            output = event["data"].get("output")
            text = _text_of(getattr(output, "content", output))
            text = text if len(text) <= 300 else text[:300] + "..."
            if name == "task":
                print(f"[智能体]返回结果:{text}")
            else:
                print(f"[工具]返回结果:{text}")


asyncio.run(run())
print()


# ============ 【对照】updates 模式写法（需配合 main_agent.stream 使用）============
# 用这套时把上面的 asyncio.run(run()) 注释掉，再取消下面的注释：
# stream = main_agent.stream(payload)   # 默认 stream_mode="updates"
# for chunk in stream:
#     # chunk -> {tools / model :{messages:[]}}
#     # {'TodoListMiddleware.after_model': None}
#     # {node_name = tools / model : state  =  {messages:[]}}
#     # node_name model / tools  state {message:[]}
#     for node_name,state in chunk.items():
#         # 有可能过程的无用节点
#         # {'TodoListMiddleware.after_model': None} 剔除
#         if not state or "messages" not in state: continue
#         # [AI / ToolsMessage]
#         messages = state.get("messages",[])
#         if len(messages) > 0:
#             # messages : [只有一个]
#             # AI / ToolsMessage
#             message = messages[-1]
#             if node_name == "model":
#                 # AIMessage = (content="",tool_calls = [{name:"task/工具名", args:{task sub.. des.. || 工具 参数名:xx} },{}])
#                 if message.content:
#                     # 调用了模型最终结果
#                     print(f"[模型]最终结果:{message.content}")
#                 else:
#                     # 调用了模型,模型决定调用工具或者子智能体
#                     tool_calls = message.tool_calls
#                     for tool_call in tool_calls:
#                         name = tool_call['name']
#                         if name == "task":
#                             # 模型决定调用子智能体
#                             sub_agent_name =  tool_call['args']['subagent_type']
#                             sub_agent_desc =  tool_call['args']['description']
#                             print(f"[模型]决定调用子智能体:{sub_agent_name},描述:{sub_agent_desc}")
#                         else:
#                             # 模型决定调用
#                             tool_name = tool_call['name']
#                             tool_args = tool_call['args']
#                             print(f"[模型]决定调用工具:{tool_name},参数:{tool_args}")
#             elif node_name == "tools":
#                 # tools 调用工具或者智能体的返回结果
#                 # ToolsMessage
#                 tool_name = message.name
#                 tool_content = message.content
#                 if tool_name == "task":
#                     # 调用子智能体
#                     print(f"[智能体]返回结果:{tool_content}")
#                 else:
#                     print(f"[工具]返回结果:{tool_content}")