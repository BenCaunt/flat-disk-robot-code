# Agent Startup: Open-Vocab Navigation Research

For work on the open-vocabulary indoor navigation research loop, use Warmhub as
the durable cross-agent memory layer:

```bash
export WARMHUB_REPO=bencaunt-2/open-vocab-nav-research-loop
wh prime
wh repo describe --json
uv run --project sim flatdisk-sim-research-warmhub status --limit 20
uv run --project sim flatdisk-sim-research-warmhub task-list --status planned --ready-only --limit 20
wh thing query --shape AgentTask --where status=running --limit 20 --json
wh thing query --shape NavEvalRun --limit 20 --json
wh thing query --shape PromptVariant --limit 20 --json
wh assertion list --shape FailureObservation --limit 20 --json
wh thing search "current open vocabulary navigation failures qwen florence exploration" --mode hybrid --json
```

Record each simulator or robot run as a `NavEvalRun`, attach generated artifacts
as `NavArtifact` records, and write `FailureObservation` assertions for failures
or suspicious successes. Keep policy inputs true to the real robot: RGB frames,
previous motion strips, IMU yaw, bounded tool results, and model memory. Hidden
THOR metadata is evaluator/debug data only.
Research configs default to `strict_model_based=true`: non-model smoke runners
and THOR-derived semantic topomap term routing are rejected unless a config
explicitly opts out for debugging. Strict `noHardcodedLabelsOrColors` results
also require a clean static prompt audit.

For planned queue work, use:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-claim \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --owner agent-name \
  --note "Starting topomap fixture build."

uv run --project sim flatdisk-sim-research-warmhub task-finish \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --status complete \
  --summary "Topomap fixture build finished; see attached output artifacts."
```

Planned tasks may include `notes.prerequisites`. Prefer `task-list
--ready-only` or the Runpod dispatcher default so trial slices wait for fixture
and preflight tasks. `task-claim` rejects incomplete prerequisites unless
`--force` is supplied.

For simple queued command execution, a worker can claim, run
`notes.commands[N]`, and finish the task in one command. Use `--all-commands`
for multi-command tasks such as topomap fixture generation:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-run-command \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --complete-exit-code 0 \
  --complete-exit-code 2 \
  --log-file sim/scratch/agent_logs/qwen-strategy-v4-topomap-fixtures.log
```

Use `task-start` only for new ad hoc work that is not already represented by a
planned `AgentTask`. Use `note` for durable scratchpad observations.
For research-loop commands, exit code `2` means the eval produced unsuccessful
navigation or structured failure data; treat it as completed evidence unless the
artifact/log output shows an infrastructure crash.

Before using exported runs for SFT/GRPO/PPO, materialize and audit Qwen tool-use
records:

```bash
uv run --project sim flatdisk-sim-prepare-qwen-tool-training \
  --input path/to/training_export/policy_dataset_v1 \
  --output-dir path/to/qwen_tool_training
```

The materializer writes Qwen-compatible multimodal SFT messages only for samples
with clean policy inputs, positive SFT weight, existing image artifacts, and
matching actor/executed actions. Evaluator rewards stay outside `messages`.

For Runpod/Linux slices, prefer the `qwen-strategy-runpod-linux-v4` queue from
`experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json`
or `scripts/runpod_open_vocab_nav_research_loop.sh`. Use
`flatdisk-sim-research-loop --preflight-only` or `PREFLIGHT_ONLY=1` to validate
Qwen endpoint and topomap fixtures before launching THOR episodes.
Set `START_QWEN_SERVER=1` on the wrapper when the pod should start its own
local vLLM server for `Qwen/Qwen3-VL-8B-Instruct` on `127.0.0.1:8000`.
The v4 fixture, preflight, and topomap-memory slice commands include conditional
`flatdisk-sim-build-semantic-topomap` guards, so fresh pods can create missing
CLIP map artifacts before running the eval without rebuilding existing maps.

To turn a planned Warmhub task into a Runpod pod launch command, use dry-run
first:

```bash
uv run --project sim flatdisk-sim-runpod-launch-task \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --start-qwen-server \
  --terminate-after 4h
```

Real launches require `runpodctl`, `RUNPOD_API_KEY`, a pushed git ref containing
the worker code, and a pod image or `WH_INSTALL_CMD` that provides the `wh` CLI.
Codex shells do not automatically load the repo root `.env`; if `.env` contains
`RUNPOD_API_KEY`, load only that variable before Runpod commands and do not print
it:

```bash
export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' .env | cut -d= -f2- | sed 's/^"//;s/"$//')"
runpodctl user
```

`runpodctl user` returning account JSON means Runpod auth is available. An
`api key not configured` error means the variable was not loaded into that
shell, not necessarily that the key is invalid.

Current handoff for GRPO training smoke, 2026-06-14:

- The repo root `.env` `RUNPOD_API_KEY` was verified with `runpodctl user` when
  loaded by the command above. Do not print the key.
- Reuse the existing Runpod pod before launching another one:
  `pbjlh2zytzulte` / `flatdisk-openloris-scene-full`, A40 46 GB, SSH
  `root@69.30.85.150 -p 22031` with
  `/Users/bencaunt/.runpod/ssh/runpodctl-ssh-key`.
- `/workspace/flat-disk-robot-code` on that pod is not a git checkout. For a
  smoke run, create a fresh `/workspace/flat-disk-robot-code-smoke` checkout of
  `codex/open-vocab-nav-research-loop` rather than modifying the existing
  directory.
- The ready local offline replay GRPO job is
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run`.
  Local planning reported `status=ready`, `sample_count=98`,
  `trainable_group_count=1`, `missing_image_count=0`, and no forbidden model
  token hits. Local runner dry-run with `--skip-dependency-check` succeeded.
- Next action is to transfer only the generated job files plus dataset-referenced
  images to the pod, run `flatdisk-sim-run-qwen-grpo-training --dry-run` with
  real dependency checks, then attempt a tiny real run. Do not claim training
  success until `qwen_grpo_training_result.json` exists.
- Expected blockers to verify on the pod: `transformers`, `trl`, `datasets`,
  `accelerate`, `peft`, and `pillow` are not installed by default; `uv` may not
  see the globally installed Torch; the generated processor load may reject
  `padding_side`; Hugging Face model download/cache state may matter.

To fan out several planned trial-slice tasks, use the dispatcher. It is also
dry-run by default and skips incomplete prerequisites unless
`--ignore-prerequisites` is supplied:

```bash
uv run --project sim flatdisk-sim-runpod-dispatch \
  --name-prefix qwen-strategy-runpod-linux-v4-run- \
  --tag trial-slice \
  --max-workers 2 \
  --start-qwen-server \
  --terminate-after 4h
```
