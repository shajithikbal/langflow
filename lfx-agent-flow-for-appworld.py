import asyncio
import os
from pathlib import Path

from dotenv import dotenv_values

from lfx.graph import Graph
from lfx.base.mcp.util import update_tools
from lfx.components.models_and_agents.agent import AgentComponent
from lfx.components.litellm.litellm_proxy import LiteLLMProxyComponent
from lfx.components.input_output.chat import ChatInput
from lfx.components.input_output.chat_output import ChatOutput
from lfx.schema.schema import InputValueRequest

import json
import re
from jinja2 import Template


async def get_available_tasks(tools_by_name: dict, dataset: str, limit: int = 10_000) -> list[str]:
    """Return the list of AppWorld task IDs for `dataset` by calling the MCP tool.

    Uses the already-loaded `list_available_tasks` StructuredTool — no JSON-RPC
    plumbing needed since we have direct in-process tool handles.
    """
    tool = tools_by_name.get("list_available_tasks")
    if tool is None:
        raise RuntimeError(
            "MCP tool `list_available_tasks` was not loaded from the server. "
            f"Available tools: {list(tools_by_name)}"
        )

    raw = await tool.ainvoke({"dataset": dataset, "limit": limit})

    # MCP returns a CallToolResult Pydantic object:
    #     CallToolResult(content=[TextContent(type='text', text='[ ...JSON... ]')], ...)
    content = getattr(raw, "content", None)
    if content:
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                return json.loads(text)

    # Fallbacks for other transports / wrappings
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, tuple) and raw:
        first = raw[0]
        if isinstance(first, str):
            return json.loads(first)
        if isinstance(first, list):
            return first

    raise RuntimeError(
        f"Unexpected return shape from list_available_tasks: "
        f"{type(raw).__name__}: {raw!r}"
    )



PROMPT_TEMPLATE = """\
Please solve AppWorld task `{task_id}` end-to-end using the MCP tools you have access to.

Workflow you must follow:

1. Call `initialize_task` with task_id="{task_id}" and experiment_name="{experiment_name}".
2. Call `get_task_info` with the same task_id to read:
     - the supervisor's name / email / phone
     - the task `instruction`
3. Repeatedly call `execute_code(task_id="{task_id}", code=<python>)` to interact with
   the AppWorld environment. The code you send is executed in a Python REPL that
   exposes the `apis` namespace (e.g. `apis.supervisor`, `apis.spotify`, `apis.api_docs`).
   Useful starters:
       print(apis.api_docs.show_app_descriptions())
       print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
       print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
4. When the task is done, call `execute_code` with code:
       apis.supervisor.complete_task(answer=<answer>)
   (omit / pass None for `answer` if the task asks for an action, not information).
5. Optionally call `check_task_completed` and `cleanup_task` when finished.

Key rules:
  - Only use the `apis` namespace inside execute_code -- no os/file/network/spotipy/etc.
  - Answers must be a bare entity / number, not a sentence (e.g. answer=10, not "ten songs").
  - For temporal queries, use [00:00:00 .. 23:59:59] of the day in question.
  - "My friends/family" means contacts in my phone app.
  - Personal info / app credentials live in the supervisor app.
  - All decisions autonomous -- never ask me to confirm.
  - For paginated APIs, walk all pages.
  - Time/date: get it from the phone app or `datetime.now()`, not from your own knowledge.

Begin now with step 1.
"""

# --- Load secrets from config file ---
#default_config = Path("~/secrets/litellm.env").expanduser()
#config_path = Path(os.environ.get("LITELLM_CONFIG", default_config))
config_path = Path("/home/sha/work/soft-innov/llm/agentic/test-time-learning/basics/litellm/.env")
if not config_path.is_file():
    raise FileNotFoundError(f"LiteLLM config file not found: {config_path}")
config = dotenv_values(config_path)

async def main():
    # --- Load MCP tools directly from the server ---
    server_name = "local-mcp"
    server_config = {
        "mode": "Streamable_HTTP",          # or "SSE" if your server uses SSE
        "url": "http://127.0.0.1:34450/mcp", # change suffix to /sse if SSE
        "headers": {},
    }
    _name, mcp_tools, _tools_by_name = await update_tools(
        server_name=server_name,
        server_config=server_config,
    )
    print(f"Loaded {len(mcp_tools)} tool(s) from MCP server: "
          f"{[t.name for t in mcp_tools]}")

    # --- Components ---
    chat_input = ChatInput()
    #chat_input.set(input_value="List the tools you have.")
    llm = LiteLLMProxyComponent()
    agent = AgentComponent()
    chat_output = ChatOutput()

    # --- Configure the LiteLLM proxy ---
    llm.set(
        api_base=config["LITELLM_BASE_URL"], #"https://ete-litellm.ai-models.vpc-int.res.ibm.com",     # your LiteLLM proxy URL (incl. /v1)
        api_key=config["LITELLM_API_KEY"], #os.environ.get("LITELLM_API_KEY", "sk-..."),   # your virtual key
        model_name="claude-haiku-4-5", #"gpt-4o-mini",                # any model your proxy routes for
        temperature=0.7,
        max_tokens=0,                            # 0 = no limit
        timeout=600,
        max_retries=2,
        stream=True,
    )

    # --- Wiring ---
    agent.set(
        input_value=chat_input.message_response,
        model=llm.build_model,                   # ← the LLM the agent uses
        tools=mcp_tools,            # plain list of StructuredTool — no edge wiring
        system_prompt="You are a helpful agent. Use MCP tools when relevant.",
        max_iterations=300, #60,     # ← AppWorld tasks need many tool-call rounds
        use_guidelines=True,             # apply guidelines from the store
        learn_guidelines_online=True,    # curate and update the store after each run
        guidelines_store_type="file",
        guidelines_file_path="/home/sha/work/soft-innov/llm/agentic/test-time-learning/frameworks/langflow/code/dev-17jun26/trials/trial-0/guidelines.json",
        #guidelines_file_path="/root/blocked.json", #"/home/sha/work/soft-innov/llm/agentic/test-time-learning/frameworks/langflow/code/dev-17jun26/trials/trial-0/guidelines.json",
    )
    chat_output.set(input_value=agent.message_response)

    graph = Graph(start=chat_input, end=chat_output)
    agent_vertex = graph.get_vertex(agent.get_id())

    DATASET_NAME = "dev" #"test_normal"            # train | dev | test_normal | test_challenge
    TASK_COUNT_LIMIT: int | None = 4        # cap how many you run this time; None = all

    all_task_ids = await get_available_tasks(_tools_by_name, DATASET_NAME, limit=10_000)
    if TASK_COUNT_LIMIT is not None:
        all_task_ids = all_task_ids[:TASK_COUNT_LIMIT]
    print(len(all_task_ids))
    print(all_task_ids)

    print(f"Got {len(all_task_ids)} task(s) from dataset={DATASET_NAME!r}.")

    #for task_id in all_task_ids:
        # ... your existing per-task body ...


    # --- Run ---
    #questions = ["What is 2 + 2?", "List the tools you have.", "Can you list me the tasks available?"]
    question = "Solve the task."
    #task_id = "82e2fac_1"
    #experiment_name="test-experiment"
    #for question in questions:
    for task_id in all_task_ids:
        #agent_vertex.update_raw_params(
        #   {"system_prompt": "You are a helpful agent. Use MCP tools when relevant and as required."},
        #    overwrite=True,
        #)
        agent_vertex.update_raw_params(
           {"system_prompt": PROMPT_TEMPLATE.format(task_id=task_id, experiment_name="test-experiment")},
            overwrite=True,
        )
        #graph = Graph(start=chat_input, end=chat_output)
        async for step in graph.async_start( #):
            inputs=InputValueRequest(
                components=["Chat Input"],
                input_value=question,
                type="chat",
                session=f"appworld-{task_id}",
            ),
        ):
            if not hasattr(step, "result_dict"):
                continue
            #print(f"--- {step.vertex.display_name} ---")
            if step.vertex.display_name not in ("Agent", "Chat Output"):
                continue
            for key, value in step.result_dict.results.items():
                # value can be a Message, a JSON wrapper, or a plain string
                text = (
                    getattr(value, "text", None)
                    or (value.data.get("text") if hasattr(value, "data") and isinstance(value.data, dict) else None)
                    or str(value)
                )
                print(f"[Q] {question}")
                print(f"[{key}] {text}")
                print()


if __name__ == "__main__":
    asyncio.run(main())
