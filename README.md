# Stock Picker — CrewAI on AWS Bedrock AgentCore

A hierarchical multi-agent crew that picks a promising company to watch in a given sector. It
searches recent news, researches the candidates, and emits a final pick with rationale — fully
managed on **AWS Bedrock AgentCore Runtime**, with **AgentCore Memory** for
short-/long-term context across sessions.

> TL;DR — You send `{"sector": "Renewable Energy"}` to the deployed runtime; a manager agent
> orchestrates three worker agents (finder → researcher → picker) and returns a written
> recommendation. Memory, LLM inference, containerization, scaling, and secrets are all
> AWS-managed.

---

## Table of contents

- [Elevator pitch](#elevator-pitch)
- [Architecture](#architecture)
- [How the crew works](#how-the-crew-works)
- [Tech stack](#tech-stack)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Local development](#local-development)
- [Secrets management](#secrets-management)
- [Deploy to AWS Bedrock AgentCore](#deploy-to-aws-bedrock-agentcore)
- [Invoke the deployed agent](#invoke-the-deployed-agent)
- [Memory model (STM + LTM)](#memory-model-stm--ltm)
- [LLMs and inference profiles](#llms-and-inference-profiles)
- [Observability](#observability)
- [Known issues / troubleshooting](#known-issues--troubleshooting)
- [Teardown](#teardown)
- [Talking points for a demo](#talking-points-for-a-demo)

---

## Summary

Three things make this project interesting:

1. **Agentic workflow, not a monolithic prompt.** A hierarchical CrewAI crew with a manager that
   delegates work to specialized worker agents (news finder, researcher, picker). Each has its
   own tools and its own LLM choice.
2. **Fully managed deployment.** No ECS, no Lambda wiring, no Docker knowledge required at
   runtime. `agentcore deploy` builds an ARM64 image in AWS CodeBuild, pushes to ECR, and
   spins up an HTTPS-accessible serverless runtime with IAM, logging, tracing, and memory
   out-of-the-box.
3. **Persistent memory across sessions.** AgentCore Memory gives the crew short-term (recent
   turns) and long-term (summarized facts) recall keyed on `actor_id + session_id`, so it
   doesn't re-recommend the same company twice.

---

## Architecture

```
┌──────────────────────────┐
│  Caller (CLI / boto3)    │
│  POST invoke_agent_runtime│
└───────────────┬──────────┘
                │ JSON payload: {sector, actor_id, session_id}
                ▼
┌───────────────────────────────────────────────────────────────┐
│               AWS Bedrock AgentCore Runtime                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Container (ARM64, CMD: python -m stock_picker_1.        │  │
│  │            agent_entrypoint)                            │  │
│  │                                                         │  │
│  │  ┌────────────────────────────────────────────────────┐ │  │
│  │  │ agent_entrypoint.invoke(payload, context)          │ │  │
│  │  │   1. load_secrets_from_aws() (Secrets Manager)     │ │  │
│  │  │   2. recall past turns from AgentCore Memory       │ │  │
│  │  │   3. run CrewAI hierarchical crew                  │ │  │
│  │  │   4. save new event back to AgentCore Memory       │ │  │
│  │  └────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────┬───────────────────────────────────────────────┘
                │
      ┌─────────┼─────────────────┬───────────────────┐
      ▼         ▼                 ▼                   ▼
┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────────┐
│ Bedrock  │ │ AgentCore    │ │ Secrets    │ │ Serper / Pushover│
│  Claude  │ │  Memory      │ │ Manager    │ │  (external APIs) │
│ (APAC    │ │ (STM + LTM)  │ │ (JSON      │ │                  │
│ inference│ │              │ │  payload)  │ │                  │
│ profiles)│ │              │ │            │ │                  │
└──────────┘ └──────────────┘ └────────────┘ └──────────────────┘
```

---

## How the crew works

A hierarchical crew (CrewAI `Process.hierarchical`) with a **Manager** delegating to three
workers, defined in `src/stock_picker_1/crew.py`:

| Agent | Role | Tools | LLM |
|---|---|---|---|
| **manager** | Plans + delegates | (delegation only) | Claude 3.5 Sonnet v2 (APAC) |
| **trending_company_finder** | Finds 2–3 trending companies in `{sector}` via live web search | `SerperDevTool` | Claude 3 Haiku (APAC) |
| **financial_researcher** | Deep-dives each candidate | `SerperDevTool` | Claude 3 Haiku (APAC) |
| **stock_picker** | Picks the best candidate, rationalizes, and pushes a notification | `PushNotificationTool` | Claude 3 Haiku (APAC) |

Task chain (`src/stock_picker_1/config/tasks.yaml`):

1. `find_trending_companies` → writes `trending_companies.json`
2. `research_trending_companies` → writes `research_report.json`
3. `pick_best_company` → sends a Pushover push + writes `decision.md`

The manager decides *when* and *to whom* to delegate; the task graph above is the structured
hand-off. Agent/task prompts live in
`src/stock_picker_1/config/{agents,tasks}.yaml`.

---

## Tech stack

| Layer | Choice |
|---|---|
| Agent framework | **CrewAI 1.14.1** (`crewai[tools]`) |
| LLM provider | **Amazon Bedrock** (Claude 3 Haiku + Claude 3.5 Sonnet v2) via APAC cross-region inference profiles |
| Search | Serper.dev (`SerperDevTool`) |
| Push notifications | Pushover (custom `PushNotificationTool`) |
| Runtime | **AWS Bedrock AgentCore Runtime** (`bedrock-agentcore` SDK, `bedrock-agentcore-starter-toolkit` CLI) |
| Memory | **AWS Bedrock AgentCore Memory** — `STM_AND_LTM` mode |
| Container build | AWS CodeBuild (ARM64), image stored in ECR |
| Secrets | AWS Secrets Manager (JSON secret loaded into `os.environ` at startup) |
| Observability | CloudWatch Logs + X-Ray + GenAI dashboard (auto-wired) |
| Packaging | `uv` + `hatchling` (standard `pyproject.toml`) |

---

## Repository layout

```
stock_picker_1/
├── .bedrock_agentcore.yaml              # AgentCore CLI config (agent, ECR, roles, memory)
├── .bedrock_agentcore/                  # Build artifacts (generated)
│   └── abhinav_stock_picker/
│       ├── Dockerfile                   # Generated ARM64 image spec
│       └── dependencies.{zip,hash}
├── .env                                 # Local dev secrets (gitignored)
├── agentcore-extra-perms.json           # IAM policy for runtime role (Secrets Manager + Memory)
├── agentcore-runtime-ecr-pull.json      # IAM policy for runtime role (ECR pull)
├── pyproject.toml                       # Python deps + project metadata
├── README.md                            # (this file)
└── src/stock_picker_1/
    ├── agent_entrypoint.py              # AgentCore entrypoint: @app.entrypoint invoke()
    ├── main.py                          # Legacy local `run_crew` entrypoint
    ├── crew.py                          # CrewAI crew, agents, tasks, Pydantic schemas
    ├── secrets.py                       # Loads JSON secret from Secrets Manager → env vars
    ├── config/
    │   ├── agents.yaml                  # Per-agent role/goal/backstory + LLM ID
    │   └── tasks.yaml                   # Task descriptions + expected_output + output_file
    └── tools/
        ├── push_tool.py                 # Pushover push-notification tool
        └── custom_tool.py               # (stub)
```

---

## Prerequisites

- **Python 3.10–3.13**, `uv` (`pip install uv` or `brew install uv`)
- **AWS account** with:
  - Bedrock model access enabled for Anthropic Claude in `ap-southeast-1`
    (Console → Bedrock → Model access)
  - Permission to create AgentCore runtimes, CodeBuild projects, ECR repos, IAM roles,
    and Secrets Manager secrets
  - AWS CLI configured (`aws configure`)
- **Serper.dev API key** (free tier): https://serper.dev/
- **(Optional) Pushover account** if you want real phone pushes from `stock_picker`

---

## Local development

```bash
# 1. Install deps
cd stock_picker_1
uv sync

# 2. Put local secrets in .env  (DO NOT COMMIT)
cat > .env <<'EOF'
SERPER_API_KEY=your_serper_key
PUSHOVER_USER=your_pushover_user_key
PUSHOVER_TOKEN=your_pushover_app_token
AWS_REGION=ap-southeast-1
EOF

# 3. Run the crew locally (no AgentCore, no memory, bedrock LLMs still used)
uv run run_crew
```

`main.py` kicks off the crew with `sector="Technology"`. Results are printed and written to
`/tmp/stock_picker_output/*` (see [known issues](#known-issues--troubleshooting) for one gotcha).

---

## Secrets management

We **do not** bake secrets into the image. Everything runs via environment variables, and in
AWS those env vars come from a single JSON blob in Secrets Manager.

1. **Create the secret** (one time):

   ```bash
   aws secretsmanager create-secret \
     --name stock-picker-1/secrets \
     --region ap-southeast-1 \
     --secret-string '{
       "SERPER_API_KEY":"...",
       "PUSHOVER_USER":"...",
       "PUSHOVER_TOKEN":"..."
     }'
   ```

2. **Tell the runtime where to look** by passing `AWS_SECRETS_ID` as an env var on deploy:

   ```bash
   uv run agentcore deploy --env AWS_SECRETS_ID=stock-picker-1/secrets
   ```

3. At startup, `src/stock_picker_1/secrets.py` fetches that secret, parses it as JSON, and
   populates `os.environ` — but it never overwrites values already set in the environment,
   so a local `.env` always wins in dev.

The execution role has `secretsmanager:GetSecretValue` scoped to this exact secret name (see
`agentcore-extra-perms.json`).

---

## Deploy to AWS Bedrock AgentCore

One-time setup (already done for this project; included for reproducibility):

```bash
# Configure the agent (writes .bedrock_agentcore.yaml)
uv run agentcore configure \
  --entrypoint src/stock_picker_1/agent_entrypoint.py \
  --name abhinav_stock_picker \
  --region ap-southeast-1 \
  --deployment-type container \
  --non-interactive
```

Attach the custom IAM policies to the auto-created runtime role (needed for Secrets Manager
reads, AgentCore Memory writes, and ECR pulls):

```bash
ROLE=AmazonBedrockAgentCoreSDKRuntime-ap-southeast-1-01bf0eaf6d

aws iam put-role-policy --role-name $ROLE \
  --policy-name stock-picker-extra-perms \
  --policy-document file://agentcore-extra-perms.json

aws iam put-role-policy --role-name $ROLE \
  --policy-name AgentCoreRuntimeEcrPull \
  --policy-document file://agentcore-runtime-ecr-pull.json
```

Then deploy (subsequent code changes only need this):

```bash
uv run agentcore deploy --env AWS_SECRETS_ID=stock-picker-1/secrets
```

What happens under the hood:

1. Zip of the source + generated Dockerfile is uploaded to S3.
2. CodeBuild (ARM64) builds the image and pushes it to ECR.
3. `CreateAgentRuntime` (or `UpdateAgentRuntime`) points the AgentCore runtime at the new image.
4. AgentCore Memory (`STM_AND_LTM`) is wired via `BEDROCK_AGENTCORE_MEMORY_ID` env var that
   the toolkit injects into the container.

Useful sibling commands:

```bash
uv run agentcore status              # show ARN, endpoint, image tag, memory id
uv run agentcore status --verbose
uv run agentcore destroy             # tear everything down
```

---

## Invoke the deployed agent

### 1. CLI (easiest)

```bash
uv run agentcore invoke '{
  "sector": "Renewable Energy",
  "actor_id": "abhinav",
  "session_id": "abhinav-2026-04-20"
}'
```

### 2. Python (boto3)

```python
import json, uuid, boto3

client = boto3.client("bedrock-agentcore", region_name="ap-southeast-1")
arn = "arn:aws:bedrock-agentcore:ap-southeast-1:478832630290:runtime/abhinav_stock_picker-sWpnJL9gXg"

resp = client.invoke_agent_runtime(
    agentRuntimeArn=arn,
    runtimeSessionId=str(uuid.uuid4()),
    payload=json.dumps({
        "sector": "Technology",
        "actor_id": "abhinav",
        "session_id": "abhinav-2026-04-20",
    }).encode("utf-8"),
    contentType="application/json",
    accept="application/json",
)
print(json.loads(resp["response"].read()))
```

### Payload contract

| Field | Required | Default | Purpose |
|---|---|---|---|
| `sector` | no | `"Technology"` | Sector the crew picks a trending company from |
| `actor_id` | no | `"stock-picker"` | AgentCore Memory actor (think: user) |
| `session_id` | no | runtime session id → `<actor_id>-default` | Conversation thread for memory recall |

Response:

```json
{
  "sector": "Renewable Energy",
  "session_id": "abhinav-2026-04-20",
  "actor_id": "abhinav",
  "result": "…final recommendation text…"
}
```

---

## Memory model (STM + LTM)

Configured in `.bedrock_agentcore.yaml` → `memory.mode: STM_AND_LTM`. The toolkit created the
memory resource `abhinav_stock_picker_mem-fM44pBFXxL` and injects its ID into the container
via `BEDROCK_AGENTCORE_MEMORY_ID`.

`agent_entrypoint.py` wraps every invoke like so:

1. **Before kickoff**: `_recall_past_context(actor_id, session_id)` calls `client.get_last_k_turns`
   (k=10) on the memory resource, formats the turns as plain text, and injects them into the
   crew inputs as `{{ past_context }}`.
2. **After kickoff**: `_save_event(...)` records two messages (the user ask, the assistant
   answer) back to the same `(actor_id, session_id)`. AgentCore's managed LTM strategies then
   asynchronously extract summaries/facts from those events so future recalls can carry
   longer-horizon context.

The effect: reusing the same `session_id` across invocations lets the picker avoid repeating
itself ("don't pick the same company twice"), even across container cold starts.

---

## LLMs and inference profiles

Bedrock in `ap-southeast-1` requires **cross-region inference profiles** for Claude 3.5+ — you
can't invoke the raw model IDs on-demand. This project uses:

| Usage | Inference profile |
|---|---|
| All worker agents | `apac.anthropic.claude-3-haiku-20240307-v1:0` |
| Manager | `apac.anthropic.claude-3-5-sonnet-20241022-v2:0` |

Change in `src/stock_picker_1/config/agents.yaml` (prefix with `bedrock/`). To see what's
available:

```bash
aws bedrock list-inference-profiles --region ap-southeast-1 \
  --query 'inferenceProfileSummaries[].{id:inferenceProfileId, name:inferenceProfileName}' \
  --output table
```

---

## Observability

Everything is wired automatically. Tail logs for a single session:

```bash
aws logs tail /aws/bedrock-agentcore/runtimes/abhinav_stock_picker-sWpnJL9gXg-DEFAULT \
  --log-stream-name-prefix "2026/04/20/[runtime-logs]" --follow \
  --region ap-southeast-1
```

GenAI dashboard (traces, token usage, tool calls, session timelines):

```
https://console.aws.amazon.com/cloudwatch/home?region=ap-southeast-1#gen-ai-observability/agent-core
```

Observability may take up to 10 minutes to show data after the first launch.

---

## Known issues / troubleshooting

### 1. `PermissionError: '/app/output/trending_companies.json'` (open)

`src/stock_picker_1/crew.py` hard-codes `output_file='output/...'` on each task:

```56:56:src/stock_picker_1/crew.py
        return Task(config=self.tasks_config['find_trending_companies'],
                     output_file='output/trending_companies.json')
```

In the container this resolves to `/app/output`, which the non-root `bedrock_agentcore` user
(uid 1000) can't write to. `tasks.yaml` already points to `/tmp/stock_picker_output/...`, but
the Python-level override in `crew.py` wins.

**Fix:** remove the `output_file=...` kwarg from each `Task(...)` in `crew.py` and let the
YAML value take effect, e.g.:

```python
@task
def find_trending_companies(self) -> Task:
    return Task(config=self.tasks_config['find_trending_companies'])
```

Alternatively, change the three hard-coded paths in `crew.py` to `/tmp/stock_picker_output/...`
as well. Then redeploy.

### 2. Manager produces `"Delegate work to coworker" arguments validation failed`

CrewAI's hierarchical process occasionally prompts Sonnet to emit the `coworker` arg as a JSON
object instead of a plain string; CrewAI retries and generally recovers. If it becomes a
blocker, either:
- switch to `Process.sequential` in `crew.py` (simpler, deterministic chain), or
- pin the manager to Claude 3.7 Sonnet (better instruction following for tool args).

### 3. `ValidationException: model ID ... with on-demand throughput isn't supported`

You're using a raw model ID (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`). Switch to the
corresponding **inference profile** ID (prefixed `apac.` or `global.`) in `agents.yaml`.

### 4. `CodeBuild is not authorized to perform: sts:AssumeRole`

IAM eventual consistency. The CodeBuild role was created microseconds ago and hasn't
propagated. Wait 30–60 s and retry. `.bedrock_agentcore.yaml` now pins the already-created
role ARN (`codebuild.execution_role`) so subsequent deploys don't try to re-create it.

### 5. `Launch failed: Package size (265 MB) exceeds 250MB limit`

That's the `direct_code_deploy` path's ZIP limit. The project is configured for
`deployment_type: container` in `.bedrock_agentcore.yaml`, which uses CodeBuild → ECR (up to
10 GB). If you ever see this, make sure the YAML has `deployment_type: container` and the
Dockerfile at `.bedrock_agentcore/abhinav_stock_picker/Dockerfile` exists.

### 6. Missing Dockerfile after manually editing `.bedrock_agentcore.yaml`

The toolkit only regenerates the Dockerfile during `agentcore configure`. If you hand-edit the
YAML (e.g. to flip `direct_code_deploy` → `container`), re-run `configure` or regenerate the
Dockerfile directly:

```bash
uv run python - <<'PY'
from pathlib import Path
from bedrock_agentcore_starter_toolkit.utils.runtime.container import ContainerRuntime
from bedrock_agentcore_starter_toolkit.utils.runtime.config import get_agentcore_directory

root = Path.cwd()
out = get_agentcore_directory(root, "abhinav_stock_picker", str(root))
out.mkdir(parents=True, exist_ok=True)
ContainerRuntime(None).generate_dockerfile(
    agent_path=root / "src/stock_picker_1/agent_entrypoint.py",
    output_dir=out, agent_name="bedrock_agentcore",
    aws_region="ap-southeast-1", enable_observability=True,
    requirements_file=None,
    memory_id="abhinav_stock_picker_mem-fM44pBFXxL",
    memory_name="abhinav_stock_picker_mem",
    source_path=str(root), protocol="HTTP", language="python",
)
PY
```

Note: the toolkit's auto-generated CMD uses `src.stock_picker_1.agent_entrypoint`; fix it to
`stock_picker_1.agent_entrypoint` (the installed package name) before deploying.

---

## Teardown

```bash
uv run agentcore destroy
```

This removes the runtime, CodeBuild project, and ECR repository (by default). Memory,
Secrets Manager secret, and IAM roles/policies are not deleted automatically — clean them up
manually if you're done:

```bash
aws secretsmanager delete-secret --secret-id stock-picker-1/secrets --force-delete-without-recovery --region ap-southeast-1
aws bedrock-agentcore delete-memory --memory-id abhinav_stock_picker_mem-fM44pBFXxL --region ap-southeast-1
aws iam delete-role-policy --role-name AmazonBedrockAgentCoreSDKRuntime-ap-southeast-1-01bf0eaf6d --policy-name stock-picker-extra-perms
aws iam delete-role-policy --role-name AmazonBedrockAgentCoreSDKRuntime-ap-southeast-1-01bf0eaf6d --policy-name AgentCoreRuntimeEcrPull
```

---

## Talking points for a demo

When you explain this to peers, lead with these:

1. **"CrewAI handles the agentic orchestration; AgentCore handles the platform."** CrewAI
   gives us agents, tools, delegation, and task graphs. AgentCore gives us the HTTPS endpoint,
   container scheduling, IAM, memory, and observability. Clean separation of concerns.

2. **"We never wrote Docker, ECS, or Lambda code."** `.bedrock_agentcore.yaml` + one `agentcore
   deploy` turns a local Python module into a production serverless runtime. CodeBuild builds
   ARM64 images in the cloud, so no local Docker needed.

3. **"Memory is keyed on `(actor_id, session_id)`."** Same session → the crew remembers what
   it picked last time. Different session → fresh context. LTM runs async in the background to
   extract long-horizon facts.

4. **"All secrets come from Secrets Manager at runtime."** The image is safe to share. Local
   devs use `.env`; the loader respects existing env vars so it's drop-in for both.

5. **"Bedrock inference profiles are regional routers."** Using `apac.anthropic.*` profiles
   gets us multi-region failover and on-demand throughput without us managing provisioned
   capacity.

6. **"IAM is scoped, not wildcarded."** The execution role has fine-grained access only to
   this project's secret ARN, this memory resource, this ECR repo, and Bedrock model
   invocation. Nothing else.

Ideas for follow-up work:
- Add a streaming protocol (AgentCore supports SSE) so the UI can show the manager's
  reasoning live.
- Swap Serper for a Bedrock Knowledge Base (internal research corpus) for a private-data play.
- Add a second crew that watches the portfolio and produces weekly rebalancing suggestions —
  reuse the same memory namespace.
