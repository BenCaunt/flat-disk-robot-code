# Open-Vocabulary Navigation Research Loop

This is the research-loop contract for general indoor navigation experiments.
The policy input path must stay true to the real robot: RGB camera frames,
previous motion strips, IMU yaw, bounded tool results, and model memory. Hidden
THOR metadata, target distance, scene objects, and target labels are allowed only
for scoring, debugging, and artifact review outside the policy directory.

## Current Loop

`flatdisk-sim-research-loop` reads a JSON experiment config, expands a trial
matrix across prompt/model/tool variants and THOR episodes, and can execute the
matrix in parallel through the existing `run_thor_harness_episode` path.

Dry-run mode writes:

- `research_loop_manifest.json`: exact trial matrix and config.
- `research_loop_summary.json`: aggregate status.
- `research_loop_report.md`: human-readable report.
- `warmhub_shapes.json`: proposed Warmhub shapes.
- `warmhub_ops.json`: Warmhub commit operations for experiments, variants,
  episode specs, runs, artifacts, assessments, and failure observations.
- `AGENTS.warmhub.md`: startup instructions for future agents.

Executed runs also write `training_export/` with policy-step JSONL,
policy-review traces, episode-rollout JSONL, rollout groups, trajectory
preference pairs, and `policy_dataset_v1/`. The policy channel contains actor
prompts, image/contact sheet paths, sanitized observations, actions, and tool
feedback. `policy_review_traces.jsonl` is the compact sub-agent review surface
for Qwen actor outputs, critic decisions, executed tools, grounding audits, and
general flags such as unstable visual-servo grounding. Hidden THOR
distance/success is kept in evaluator-only reward/label channels for offline
ranking, SFT filtering, PPO, or GRPO; it is never inserted into model inputs or
policy-review traces. `flatdisk-sim-prepare-qwen-tool-training` can also
materialize guard-replaced actor actions as Qwen action-preference pairs, where
the executed safe action is the chosen target and the rejected actor action is
kept out of accepted SFT. It also writes `qwen_dpo_messages.jsonl` with explicit
`prompt`, `chosen`, `rejected`, and `images` columns for a future VLM
preference-training worker.

Seed config:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_prompt_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Topomap-memory comparison config:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Runpod/Linux generated strategy config:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Generated strategy sweeps can be created from an existing config:

```bash
uv run --project sim flatdisk-sim-generate-nav-strategy-sweep \
  --base-config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep_runpod_linux.json \
  --output experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --experiment-id open_vocab_nav_qwen_strategy_runpod_linux_v1
```

This emits generic Qwen strategies for baseline behavior, viewpoint coverage,
evidence exploitation, failure recovery, and optional CLIP-backed topomap image
memory. Generated strategies are covered by tests that audit static prompt
context for object/color terms.

Remove `--dry-run` to launch THOR. Add `--parallelism N` to override config
concurrency. Add `--preflight-endpoints` to check OpenAI-compatible Qwen
endpoints before launching THOR and record unavailable endpoints as structured
failed trials. Add `--preflight-only` to perform endpoint/topomap checks and
stop before launching THOR.

## Current Tool Surface

Qwen can choose bounded motion, Florence/GroundingDINO phrase-grounded visual
servo, `wait`, `stop`, and `query_topomap_memory`. The topomap memory tool is
non-motion and opt-in: variants may set `topomap_memory_map_dir` and
`topomap_memory_use_clip` to let Qwen query a saved semantic topomap with the
latest RGB frame and text goal. The model-facing result contains image-match
scores, route node ids, and a contact sheet; it omits poses, scene metadata,
object metadata, target distance, and hidden THOR state.

Qwen variants use `critic_mode: none` by default in the active configs. The
actor therefore owns bounded tool selection and can call `stop` to assert
completion without a local rule-based critic replacing the action. The old
`SafetyCriticRunner` remains available only for explicit `critic_mode: safety`
baseline/debug runs, and `critic_mode: same-model` can be used for a separate
model-based critic ablation.

Strict policy runs should prefer CLIP-backed topomaps. The
`topomap_memory_allow_semantic_terms` switch is reserved for debugging or maps
whose semantic terms are known to come from a real-robot-compatible process.
`topomap_memory_map_dir` may use `{episode}` so one sweep can compare per-scene
maps such as `sim/scratch/semantic_topomaps/living_room_sofa_clip`.

## Warmhub

Warmhub is the cross-agent memory layer. Its docs describe shapes, things,
assertions, and wrefs as the core data model, with CLI writes through `wh commit
submit`. New agents should run `wh prime` at startup, then query recent
`AgentTask`, `NavEvalRun`, `PromptVariant`, and `FailureObservation` records
before choosing work.

Default repo target:

```bash
export WARMHUB_REPO=bencaunt-2/open-vocab-nav-research-loop
wh prime
wh repo describe --json
uv run --project sim flatdisk-sim-research-warmhub status --limit 20
uv run --project sim flatdisk-sim-research-warmhub task-list --status planned --ready-only --limit 20
wh thing query --shape AgentTask --where status=running --limit 20 --json
wh thing query --shape AgentTask --where status=planned --limit 20 --json
wh thing query --shape NavEvalRun --limit 20 --json
wh thing query --shape PromptVariant --limit 20 --json
wh assertion list --shape FailureObservation --limit 20 --json
```

For a compact queue and eval triage view, use:

```bash
uv run --project sim flatdisk-sim-research-warmhub status --limit 20
uv run --project sim flatdisk-sim-research-warmhub status \
  --related-experiment open_vocab_nav_qwen_strategy_runpod_linux_v1 \
  --json
```

The status command summarizes task counts, recent `NavEvalRun` records,
failure observations, sub-agent results, and recommended next actions.
It flags running tasks whose `updatedAt` is older than four hours by default;
use `--stale-running-after-s 0` to disable that check, or lower the threshold
when auditing short pod jobs.
Planned tasks may declare `notes.prerequisites`; workers should use
`task-list --ready-only` or the Runpod dispatcher default so trial slices are
not claimed before fixture/preflight tasks are complete. `task-claim` also
rejects incomplete prerequisites unless `--force` is supplied.

The research loop does not write to Warmhub unless `--commit-warmhub` is passed.
Use `--init-warmhub-repo` to create the repo and missing shapes before submit.
Agents should claim planned queue work before running it, then finish the task
with evidence artifacts:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-list \
  --status planned \
  --ready-only \
  --limit 20

uv run --project sim flatdisk-sim-research-warmhub task-claim \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --owner agent-name \
  --note "Starting topomap fixture build."

uv run --project sim flatdisk-sim-research-warmhub note \
  --author agent-name \
  --about NavExperiment/open_vocab_nav_qwen_prompt_sweep_v1 \
  --note "Observed repeated final-goal servo without visible target evidence." \
  --tag qwen --tag failure

uv run --project sim flatdisk-sim-research-warmhub task-finish \
  --task inspect-failures-001 \
  --agent agent-name \
  --status complete \
  --summary "Failure mode logged; next action is an exploration-biased prompt."
```

For unattended local or pod-side workers, execute queued commands with the full
claim/run/finish lifecycle. Multi-command fixture tasks should use
`--all-commands` so every map build and validation command finishes before the
task is marked complete:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-run-command \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --complete-exit-code 0 \
  --complete-exit-code 2 \
  --log-file sim/scratch/agent_logs/qwen-strategy-v4-topomap-fixtures.log \
  --dry-run

uv run --project sim flatdisk-sim-research-warmhub task-run-command \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --complete-exit-code 0 \
  --complete-exit-code 2 \
  --log-file sim/scratch/agent_logs/qwen-strategy-v4-topomap-fixtures.log \
  --evidence-artifact sim/scratch/semantic_topomaps/living_room_sofa_clip
```

Use `task-start` only for ad hoc work that is not already represented by a
planned `AgentTask`.
Research-loop exit code `2` is accepted for queued worker tasks because it means
the eval produced unsuccessful navigation or structured failure data; worker
failure should be reserved for crashes, missing artifacts, or command errors.

When a failure trace shows `visual_servo_object` returned `no_detection` on
visually plausible frames, run the detector doctor on saved policy frames before
rerunning a full slice:

```bash
uv run --project sim --with torch --with torchvision --with transformers --with timm --with einops \
  flatdisk-sim-detector-doctor \
  --image path/to/frame.jpg \
  --prompt "visible object phrase" \
  --detector grounding-dino \
  --output-dir sim/scratch/detector_doctor/run-id
```

The doctor writes JSON, Markdown, and detector overlays so agents can compare
Florence/GroundingDINO-style backends without using hidden simulator metadata.

To create a Warmhub work queue from a research-loop config:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-plan-config \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --plan-id qwen-strategy-runpod-linux-v4 \
  --owner unassigned \
  --tag qwen --tag runpod --tag generated-strategies \
  --include-slice-tasks
```

For Runpod/Linux workers, use the v4 generated-strategy queue above. The
topomap-fixture task notes include concrete `flatdisk-sim-build-semantic-topomap`
commands for each episode and `flatdisk-sim-research-loop --preflight-only`
validation commands. The v4 fixture, preflight, and topomap-memory slice
commands include conditional map-build guards, so fresh pods can create missing
CLIP map artifacts before running the eval without rebuilding existing maps.
The generated training-review task also runs
`flatdisk-sim-prepare-qwen-tool-training` before
`flatdisk-sim-nav-training-readiness`, so WarmHub readiness assertions include
accepted Qwen SFT counts, Qwen guard-replacement preferences, and
`qwen_dpo_messages.jsonl` handoff counts.
The downstream `qwen-dpo-train-plan` task runs
`flatdisk-sim-plan-qwen-dpo-training` to validate the DPO handoff and write a
`qwen_dpo_training_job.json` plus a generated TRL script. That task is a
training handoff only; actual GPU fine-tuning should run in a later worker with
an isolated TRL/Transformers/Accelerate/PEFT environment. The following
`qwen-dpo-train-worker` task runs `flatdisk-sim-run-qwen-dpo-training`, which
checks the manifest, dataset, script, and training packages before executing
the generated `accelerate launch ...` command.
The sibling `qwen-grpo-train-plan` task runs
`flatdisk-sim-prepare-qwen-grpo-training` to materialize grouped rollout
candidates for future GRPO/PPO work. It marks only trajectories whose actor
actions were actually executed as trainable and keeps evaluator rewards outside
model-facing messages. The following `qwen-grpo-job-plan` task turns that
handoff into a `qwen_grpo_training_job.json`, `qwen_grpo_trl_dataset.jsonl`,
and generated TRL script. The worker task runs
`flatdisk-sim-run-qwen-grpo-training`; this is an offline replay/proxy GRPO
handoff, not a claim that Qwen is receiving live simulator rewards yet.

## Runpod Worker

The pod-side worker wrapper is:

```bash
TASK_ID=qwen-strategy-runpod-linux-v4-run-qwen_topomap_memory-living_room_sofa \
CONFIG=experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
VARIANT=qwen_topomap_memory \
EPISODE=living_room_sofa \
scripts/runpod_open_vocab_nav_research_loop.sh
```

It writes `/workspace/open_vocab_nav_research_loop.log`,
`/workspace/open_vocab_nav_research_loop.exit`, and timestamped output
directories under `/workspace/outputs/open_vocab_nav_research_loop` by default.
If `TASK_ID` is set, it claims the Warmhub task at startup and writes a
`SubAgentResult` on exit.

The wrapper defaults to the Runpod/Linux config, `UV_EXTRAS=thor`, and
`UV_WITH=torch,transformers`, avoiding the local MLX harness dependency. Serve
Qwen through a Linux-compatible OpenAI-style endpoint such as vLLM on
`127.0.0.1:8000`, set `START_QWEN_SERVER=1` to have the wrapper start that
endpoint with `scripts/runpod_start_qwen_vllm.sh`, or override `qwen_endpoint`
in a copied config. For v4 queued commands, topomap fixtures under
`sim/scratch/semantic_topomaps/{episode}_clip` are built automatically when
missing before preflight or topomap-memory slices run. Set `PREFLIGHT_ONLY=1`
for a validation-only pod job. The Qwen startup path is opt-in because it can
install vLLM and download model weights.

To create a Runpod launch command from a planned Warmhub task, use the
dry-run-first launcher:

```bash
uv run --project sim flatdisk-sim-runpod-launch-task \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --start-qwen-server \
  --qwen-vllm-extra-args "--max-model-len 8192" \
  --terminate-after 4h \
  --evidence-artifact /workspace/outputs/open_vocab_nav_research_loop
```

The launcher emits a `runpodctl pod create` command using Runpod's documented
`--image`, `--gpu-id`, `--env`, and `--docker-args` flags. Add `--launch` only
after the generated command looks right, `RUNPOD_API_KEY` is configured for
pod management, the git ref is pushed, and the pod image can run the `wh` CLI
or `WH_INSTALL_CMD` installs it. By default, real launch is refused from a dirty
worktree because the pod clones `origin` and would otherwise miss local worker
changes.

Local Codex shells do not automatically load the repo root `.env`. If `.env`
contains `RUNPOD_API_KEY`, load only that variable before Runpod commands and do
not print it:

```bash
export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' .env | cut -d= -f2- | sed 's/^"//;s/"$//')"
runpodctl user
```

`runpodctl user` returning account JSON means Runpod auth is available. If it
says `api key not configured`, the variable was not loaded into that shell.

### 2026-06-14 GRPO Smoke Handoff

Use the existing pod before spending time launching a second one. Verified
Runpod state:

- Pod id `pbjlh2zytzulte`, name `flatdisk-openloris-scene-full`.
- Image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`.
- GPU `NVIDIA A40`, about 46 GB VRAM.
- SSH:

```bash
ssh -i /Users/bencaunt/.runpod/ssh/runpodctl-ssh-key root@69.30.85.150 -p 22031
```

The pod's `/workspace/flat-disk-robot-code` directory is not a git checkout.
Create a fresh smoke checkout instead of trying to recover that directory:

```bash
ssh -i /Users/bencaunt/.runpod/ssh/runpodctl-ssh-key root@69.30.85.150 -p 22031 \
  'cd /workspace && git clone https://github.com/BenCaunt/flat-disk-robot-code.git flat-disk-robot-code-smoke && cd flat-disk-robot-code-smoke && git checkout codex/open-vocab-nav-research-loop && git rev-parse --short HEAD'
```

Local GRPO artifacts that are ready to smoke:

- Handoff:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_training/bathroom_cross_run/qwen_grpo_training_manifest.json`
- Job:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/qwen_grpo_training_job.json`
- Dataset:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/qwen_grpo_trl_dataset.jsonl`
- Generated script:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/train_qwen_grpo_trl.py`

The local planner reported `status=ready`, `sample_count=98`,
`trainable_group_count=1`, `missing_image_count=0`, and no forbidden model-token
hits. The local runner dry-run succeeded with:

```bash
PYTHONPATH=/tmp/codex_no_readline:sim/src uv run --project sim --extra dev \
  flatdisk-sim-run-qwen-grpo-training \
  --job sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run \
  --dry-run \
  --skip-dependency-check
```

To transfer a minimal artifact bundle, include only the generated job files and
the image paths referenced by `qwen_grpo_trl_dataset.jsonl`:

```bash
python - <<'PY'
import json
from pathlib import Path

dataset = Path("sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/qwen_grpo_trl_dataset.jsonl")
files = {
    "sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/qwen_grpo_training_job.json",
    "sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/qwen_grpo_trl_dataset.jsonl",
    "sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/train_qwen_grpo_trl.py",
}
for line in dataset.read_text().splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    files.update(record.get("image_paths", []))
Path("/tmp/qwen_grpo_smoke_files.txt").write_text("\n".join(sorted(files)) + "\n")
print(len(files))
PY
tar -czf /tmp/qwen_grpo_bathroom_cross_run_minimal.tgz -T /tmp/qwen_grpo_smoke_files.txt
scp -P 22031 -i /Users/bencaunt/.runpod/ssh/runpodctl-ssh-key \
  /tmp/qwen_grpo_bathroom_cross_run_minimal.tgz \
  root@69.30.85.150:/workspace/
```

Extract it into `/workspace/flat-disk-robot-code-smoke`, then run the real
dependency-checking dry-run before any training:

```bash
uv run --project sim --extra dev \
  --with accelerate --with datasets --with peft --with pillow --with transformers --with trl \
  flatdisk-sim-run-qwen-grpo-training \
  --job sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run \
  --dry-run
```

The one-step smoke completed on 2026-06-14 in
`/workspace/flat-disk-robot-code-smoke-20260614-grpo` using
`/workspace/flatdisk-grpo-venv`:

- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `626.335` seconds.
- Launch used `--max-steps 1`, `--num-generations 2`, and
  `--max-completion-length 64`.
- Adapter:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/adapter/adapter_model.safetensors`
  at about 87 MB.
- Training reported `21,823,488` trainable params out of `8,788,947,184`
  total params, about `0.2483%`.

The first longer capped run also completed on 2026-06-14 in
`/workspace/flat-disk-robot-code-train-20260614-grpo4`, from pushed commit
`70df4be`:

- Job:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48/qwen_grpo_training_job.json`
- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `1783.814` seconds.
- Launch used `--max-steps 4`, `--num-generations 2`, and
  `--max-completion-length 48`.
- Adapter:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48/adapter/adapter_model.safetensors`
  at about 87 MB.
- Training reported train runtime about `1551` seconds, train loss `0.01178`,
  `2.654e+05` tokens, reward mean `-0.4789`, reward std `0.1339`, and step time
  about `382.7` seconds.
- Mean completion length was `47.25`, max completion length was `48`, and
  clipped ratio was `0.9688`; this is a useful signal that the next run should
  improve completion formatting or reward/data before scaling.

Two logged one-step follow-up runs isolated and fixed the formatting issue:

- Commit `0961739` added `adapter/completion_samples.jsonl`. The run in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-log1` completed with
  `duration_s=644.804`, but `completion_samples.jsonl` showed `0/8` parsed
  actions. The model usually began with a plausible action JSON object, then
  continued into `thought` or `grounding_audit`, so the object was truncated
  before closing.
- Commit `6b4eaac` appends a generic `GRPO_RESPONSE_CONTRACT` to training
  prompts. It tells the model to output only one compact JSON object of the form
  `{"action":{"tool":"<tool_name>","args":{...}}}` and then stop. This contract
  is generic; it does not introduce object or color names.
- The comparison run in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-contract1` completed with
  status `complete`, return code `0`, and `duration_s=675.898`. It logged 8
  completion samples: `8/8` parsed actions, `5/8` exact reference actions, no
  markdown fences, and TRL `completions/clipped_ratio=0`.
- Future result manifests include `completion_log_jsonl`,
  `completion_log_sample_count`, and `completion_log_metrics` so this quality
  check can be automated in the research loop.

A larger post-fix 4-step run then completed in
`/workspace/flat-disk-robot-code-train-20260614-grpo-contract4`, from commit
`98b962a`:

- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_action_contract/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `1907.898` seconds.
- `completion_log_sample_count` was `32`.
- `completion_log_metrics`: `32/32` parsed actions, `9/32` exact reference
  actions, `0` markdown fences, `0` truncated texts, and mean completion length
  `91.75` characters.
- TRL metrics: train runtime about `1673` seconds, train loss `-0.01928`,
  reward mean `-0.132`, reward std `0.1579`, and
  `completions/clipped_ratio=0`.

That run showed the format problem is fixed, while action-choice quality still
needs work. One important reward-shaping issue was identified afterward: parsed
but non-reference actions could still receive a positive reward when the source
trajectory reward was positive. The reward function now caps non-reference
parsed actions at a negative reward and invalid JSON at a stronger negative
reward before scaling.

A reward-cap 4-step run then completed in
`/workspace/flat-disk-robot-code-train-20260614-grpo-rewardcap4`, from commit
`6f93b6c`:

- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_reward_cap/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `1919.723` seconds.
- `completion_log_sample_count` was `32`.
- `completion_log_metrics`: `32/32` parsed actions, `12/32` exact reference
  actions, `0` markdown fences, `0` truncated texts, and mean completion length
  `92.938` characters.
- TRL metrics: train runtime about `1673` seconds, train loss `-0.01638`,
  reward mean `-0.1365`, reward std `0.1809`, and
  `completions/clipped_ratio=0`.
- Manual comparison against the pre-cap 4-step run showed positive rewards for
  non-reference actions dropped from `5` to `0`, while exact reference actions
  improved from `9/32` to `12/32`.

A generic tool/argument reward-shaping 4-step run then completed in
`/workspace/flat-disk-robot-code-train-20260614-grpo-toolreward4`, from commit
`1ce438b`:

- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_reward/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `1888.442` seconds.
- `completion_log_sample_count` was `32`.
- `completion_log_metrics`: `32/32` parsed actions, `12/32` exact reference
  actions, `17/32` same-tool matches, mean argument-match fraction `0.4375`,
  `0` positive non-reference rewards, `0` markdown fences, `0` truncated texts,
  and mean completion length `92.406` characters.
- TRL metrics: train runtime about `1647` seconds, train loss `-0.01663`,
  reward mean `-0.1392`, reward std `0.1748`, and
  `completions/clipped_ratio=0`.
- This run did not improve exact action rate over the reward-cap run on the
  tiny four-step comparison. It did verify the new generic partial-credit reward
  surface: same-tool and argument-key matches receive graded negative rewards,
  while non-reference actions still never receive positive reward.

A reference-tool-balanced 4-step run then completed in
`/workspace/flat-disk-robot-code-train-20260614-grpo-toolbalanced4`, from commit
`deef420`:

- Planner change: `--balance-reference-tools` duplicates underrepresented
  reference tool-family samples with `balance_original_sample_id` provenance and
  records before/after counts in `dataset_action_audit`.
- Dataset shift: the bathroom handoff changed from `98` samples
  (`visual_servo_object=56`, `turn_by_angle=28`, `check_object_grounding=13`,
  `stop=1`) to `168` samples (`visual_servo_object=56`, `turn_by_angle=56`,
  `check_object_grounding=52`, `stop=4`) using `--max-balance-multiplier 4`.
- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_balanced/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `1695.164` seconds.
- `completion_log_sample_count` was `32`.
- `completion_log_metrics`: `32/32` parsed actions, `12/32` exact reference
  actions, `20/32` same-tool matches, mean argument-match fraction `0.489583`,
  `0` positive non-reference rewards, `0` markdown fences, `0` truncated texts,
  and mean completion length `84.938` characters.
- TRL metrics: train runtime about `1467` seconds, train loss `-0.01278`,
  reward mean `-0.1391`, reward std `0.1333`, and
  `completions/clipped_ratio=0`.
- Compared with the unbalanced shaped-reward run, exact action rate stayed
  `12/32`, same-tool matches improved from `17/32` to `20/32`, mean argument
  match improved from `0.4375` to `0.489583`, and completion length shortened.
  This suggests balancing helps action family selection, but exact argument
  choice and broader data coverage are still limiting factors.

A reference-tool-balanced 8-step diagnostic then completed in
`/workspace/flat-disk-robot-code-train-20260614-grpo-toolbalanced8`, from
commit `ea4d3fd`:

- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_8step_cap48_tool_balanced/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `3212.117` seconds.
- `completion_log_sample_count` was `64`.
- `completion_log_metrics`: `64/64` parsed actions, `25/64` exact reference
  actions, `36/64` same-tool matches, mean argument-match fraction `0.455729`,
  `0` positive non-reference rewards, `0` markdown fences, `0` truncated texts,
  and mean completion length `87.797` characters.
- TRL metrics: train runtime about `2973` seconds, train loss `-0.01136`,
  reward mean `-0.1544`, reward std `0.1401`, step time `366.6` seconds, and
  `completions/clipped_ratio=0`.
- The remote LoRA adapter saved successfully under
  `adapter/adapter_model.safetensors`; the file is about 87 MB. The top-level
  LoRA adapter files were copied back locally under the same job's `adapter/`
  directory; the optimizer checkpoint was left on the pod.
- Compared with the balanced 4-step run, exact rate only moved from `12/32`
  (`0.375`) to `25/64` (`0.390625`), while same-tool rate dropped from `0.625`
  to `0.5625` and mean argument match dropped from `0.489583` to `0.455729`.
  This says longer training alone is not yet enough.
- The main observed wrong-action pattern is now clear turn overuse. Expected
  tools in the 64 logged samples were `visual_servo_object=20`,
  `turn_by_angle=26`, and `check_object_grounding=18`; parsed tools were
  `visual_servo_object=8`, `turn_by_angle=48`, and `check_object_grounding=8`.
  Exact-by-tool was `visual_servo_object=4/20`, `turn_by_angle=15/26`, and
  `check_object_grounding=6/18`.

An exact-reference-action bonus ablation then completed in
`/workspace/flat-disk-robot-code-train-20260614-grpo-exactbonus4`, from commit
`987ec94`:

- Planner change: `--zero-reward-exact-action-bonus` adds a generic opt-in
  bonus only for exact reference actions whose source `candidate_step_reward` is
  present and zero. Missing source rewards still cannot produce positive exact
  rewards, and non-reference actions remain non-positive. The bonus value and
  reference action remain out of the prompt/response contract.
- Tests: `sim/tests/test_qwen_grpo_training.py` covers the planner audit,
  generated training script, prompt non-leakage, generic tool names, missing
  reward handling, and the non-reference positive reward cap.
- Result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_balanced_exact_bonus005/qwen_grpo_training_result.json`
- Status `complete`, return code `0`, duration `1626.631` seconds.
- `completion_log_sample_count` was `32`.
- `completion_log_metrics`: `32/32` parsed actions, `11/32` exact reference
  actions, `19/32` same-tool matches, mean argument-match fraction `0.458333`,
  `0` positive non-reference rewards, `0` markdown fences, `0` truncated texts,
  and mean completion length `84.688` characters.
- TRL metrics: train runtime about `1397` seconds, train loss `-0.01657`,
  reward mean `-0.1313`, reward std `0.1537`, step time `344.5` seconds, and
  `completions/clipped_ratio=0`.
- The remote LoRA adapter saved successfully under `adapter/adapter_model.safetensors`.
  The top-level adapter files were copied back locally under the same job's
  `adapter/` directory; the optimizer checkpoint was left on the pod.
- Compared with the no-bonus balanced 4-step run, exact matches dropped from
  `12/32` to `11/32`, same-tool matches dropped from `20/32` to `19/32`, and
  mean argument match dropped from `0.489583` to `0.458333`. This says the
  simple zero-reward exact-action bonus is not the next scaling lever.
- Turn overuse persisted. Expected tools in the 32 logged samples were
  `visual_servo_object=4`, `turn_by_angle=14`, and `check_object_grounding=14`;
  parsed tools were `visual_servo_object=5`, `turn_by_angle=20`, and
  `check_object_grounding=7`. Exact-by-tool was `visual_servo_object=0/4`,
  `turn_by_angle=6/14`, and `check_object_grounding=5/14`.

### Held-Out Completion Eval Harness

The GRPO job module now includes a held-out completion-eval path:

- `flatdisk-sim-plan-qwen-grpo-completion-eval` plans an eval job from an
  existing `qwen_grpo_training_job.json`.
- `flatdisk-sim-run-qwen-grpo-completion-eval` runs or dry-runs the planned
  eval job and writes `qwen_grpo_completion_eval_result.json`.
- The planner writes `qwen_grpo_completion_eval_dataset.jsonl` and
  `eval_qwen_grpo_completions.py`. The generated script loads base Qwen and
  optionally `PeftModel.from_pretrained(...)` for an explicit adapter path; it
  does not create a new trainable LoRA wrapper.
- The eval script sends only `prompt_messages` and loaded images into the
  processor. `reference_action_json`, `reference_action_canonical`,
  `candidate_step_reward`, and related reward fields remain evaluator-only
  sidecars used after generation for scoring.
- Completion logs reuse the existing generic action metrics and now include
  per-tool expected/parsed/exact/tool-match counts, which makes turn overuse
  visible without an ad hoc parser.

Local smoke:

```bash
PYTHONPATH=/tmp/codex_no_readline:sim/src uv run --project sim --extra dev \
  flatdisk-sim-plan-qwen-grpo-completion-eval \
  --training-job sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_balanced/qwen_grpo_training_job.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop/qwen_grpo_completion_evals/bathroom_cross_run_lora_4step_cap48_tool_balanced_base_eval_smoke \
  --max-samples 4 \
  --sample-stride 8 \
  --max-new-tokens 48 \
  --fail-on-not-ready
```

The planner reported `status=ready`, `sample_count=4`,
`missing_image_count=0`, `forbidden_model_token_hits=[]`, and
`sidecar_prompt_leak_hits=[]`. A runner dry-run with
`flatdisk-sim-run-qwen-grpo-completion-eval --dry-run --skip-dependency-check`
also succeeded. This smoke verified packaging and leakage checks; it did not
run real Qwen generation.

Real Runpod comparison:

- Remote checkout:
  `/workspace/flat-disk-robot-code-eval-20260615-qwen-completions` at commit
  `376600e`.
- Local copied artifacts:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_completion_evals/stride5_24`
- Eval slice: balanced 4-step GRPO prompt dataset,
  `--max-samples 24 --sample-stride 5 --max-new-tokens 48`.
- Expected tools: `check_object_grounding=8`, `turn_by_angle=8`,
  `visual_servo_object=7`, `stop=1`.
- Models/adapters evaluated: base `Qwen/Qwen3-VL-8B-Instruct`,
  `lora4_toolbalanced`, `lora8_toolbalanced`, and `lora4_exactbonus005`.
- All four runs completed with return code `0`.

All four runs produced identical aggregate metrics:

- `24/24` parsed actions.
- `6/24` exact reference actions.
- `15/24` same-tool matches.
- Mean argument-match fraction `0.423611`.
- `0` positive non-reference rewards.
- Parsed tools: `check_object_grounding=4`, `turn_by_angle=16`,
  `visual_servo_object=3`, `stop=1`.
- Exact-by-tool: `check_object_grounding=3/8`, `turn_by_angle=1/8`,
  `visual_servo_object=1/7`, `stop=1/1`.
- Completion text digest: `e31e4973e7bb5037` for base and every adapter.
- Byte-level completion text diffs versus base: `0` for each adapter.

Interpretation: the current LoRA adapters do not affect deterministic greedy
action completions on this eval slice. The persistent behavior is still
`turn_by_angle` overuse: expected `turn_by_angle=8`, parsed `turn_by_angle=16`.
Before spending another long run on GRPO scale, add an adapter-effect sanity
check such as logit comparison, adapter merge/load verification, or a small
overfit probe where the adapter must visibly change completions.

### Adapter-Effect Logit Check

The GRPO job module now includes a PEFT adapter-effect diagnostic for the GRPO
eval slice:

- `flatdisk-sim-plan-qwen-grpo-adapter-effect` plans a check from an existing
  `qwen_grpo_completion_eval_job.json`.
- `flatdisk-sim-run-qwen-grpo-adapter-effect` runs or dry-runs the planned
  check and writes `qwen_grpo_adapter_effect_result.json`.
- The planner copies the same prompt/image records used by completion eval into
  `qwen_grpo_adapter_effect_dataset.jsonl`, optionally with
  `--max-samples`, `--sample-offset`, and `--sample-stride`; it checks image
  existence, prompt sidecar leaks, and PEFT adapter directory completeness
  before launch.
- The generated `check_qwen_grpo_adapter_effect.py` loads base Qwen once,
  wraps it with `PeftModel.from_pretrained(...)`, compares next-token logits
  with `model.disable_adapter()` versus adapter-enabled inference on the same
  processor inputs, and appends rows to `adapter_effect_samples.jsonl` after
  each sample.
- Metrics include nonzero logit-delta rate, max/mean absolute logit delta,
  L2 logit delta, KL in both directions, top-1 changed rate, top-k Jaccard,
  and per-expected-tool counts. Expected tools are attached only after
  inference for reporting.

Local validation:

```bash
PYTHONPATH=/tmp/codex_no_readline:sim/src uv run --project sim --extra dev \
  flatdisk-sim-plan-qwen-grpo-adapter-effect \
  --completion-eval-job sim/scratch/open_vocab_nav_research_loop/qwen_grpo_completion_evals/stride5_24/base \
  --output-dir sim/scratch/open_vocab_nav_research_loop/qwen_grpo_adapter_effect_checks/stride5_24_lora8_toolbalanced_local_plan \
  --adapter-path sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_8step_cap48_tool_balanced/adapter \
  --top-k 5 \
  --fail-on-not-ready
```

This local planner reported `status=ready`, `sample_count=24`,
`missing_image_count=0`, `adapter_path_blockers=[]`,
`forbidden_model_token_hits=[]`, and `sidecar_prompt_leak_hits=[]`. A runner
dry-run also succeeded, and the generated script passed `py_compile`. Unit
coverage for the planner, blockers, dry-run, fake execution, prompt-leak
boundary, and metric aggregation is in `sim/tests/test_qwen_grpo_training.py`.

RunPod smoke on `lora8_toolbalanced`:

- Remote checkout:
  `/workspace/flat-disk-robot-code-eval-20260615-qwen-completions` at commit
  `0839be9`.
- Adapter:
  `/workspace/flat-disk-robot-code-train-20260614-grpo-toolbalanced8/sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_8step_cap48_tool_balanced/adapter`.
- Local copied artifacts:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_adapter_effect_checks/stride5_24_lora8_toolbalanced_remote_smoke2`.
- Slice: `--max-samples 2 --top-k 5`, expected tools
  `check_object_grounding=1`, `stop=1`.
- Result: `2/2` samples had nonzero logit deltas; top-1 token changed on
  `0/2` samples.
- Aggregate metrics: mean max-absolute logit delta `0.487304688`,
  max max-absolute delta `0.5625`, mean absolute logit delta `0.075996335`,
  mean L2 delta `37.379156112`, mean KL adapter-from-base `0.000000925`,
  mean top-k Jaccard `0.833333333`.
- The active PEFT adapter name was `default`, with `21,823,488` adapter
  parameters. Both checked prompts still had the same top-1 next token
  (`{"`) with and without the adapter.

Interpretation: adapter loading is working and the lora8 adapter measurably
changes logits, but the changes are too small or too off-target to change
greedy action completions on this small smoke. The next training/debug step
should focus on whether the learning signal shifts action/tool tokens enough
under the response contract, not on PEFT path loading.

### Action-Likelihood Teacher-Forced Check

The GRPO job module now includes a teacher-forced action-likelihood diagnostic:

- `flatdisk-sim-plan-qwen-grpo-action-likelihood` plans a check from an
  existing `qwen_grpo_completion_eval_job.json`.
- `flatdisk-sim-run-qwen-grpo-action-likelihood` runs or dry-runs the planned
  check and writes `qwen_grpo_action_likelihood_result.json`.
- The planner copies a selected completion-eval prompt/image slice into
  `qwen_grpo_action_likelihood_dataset.jsonl`, with `--max-samples`,
  `--sample-offset`, and `--sample-stride`.
- The generated `score_qwen_grpo_action_likelihood.py` builds the same Qwen
  prompt/images as completion eval, appends only compact
  `{"action":{"tool":"...","args":{...}}}` JSON as a teacher-forced assistant
  target, and compares base-disabled versus adapter-enabled logprobs using the
  same loaded PEFT model.
- Metrics include target mean logprob delta, target NLL delta, first target
  token logprob delta, tool-span logprob delta, tool-span found rate, and
  per-expected-tool aggregates. Reward sidecars and hidden evaluator fields
  remain excluded from prompt messages.

Local validation:

```bash
PYTHONPATH=/tmp/codex_no_readline:sim/src uv run --project sim --extra dev \
  flatdisk-sim-plan-qwen-grpo-action-likelihood \
  --completion-eval-job sim/scratch/open_vocab_nav_research_loop/qwen_grpo_completion_evals/stride5_24/base \
  --output-dir sim/scratch/open_vocab_nav_research_loop/qwen_grpo_action_likelihood_checks/stride5_24_lora8_toolbalanced_local_smoke2 \
  --adapter-path sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_8step_cap48_tool_balanced/adapter \
  --max-samples 2 \
  --fail-on-not-ready
```

This planner reported `status=ready`, `sample_count=2`,
`missing_image_count=0`, `empty_target_count=0`,
`adapter_path_blockers=[]`, `forbidden_model_token_hits=[]`, and
`sidecar_prompt_leak_hits=[]`. A runner dry-run succeeded, the generated script
passed `py_compile`, `sim/tests/test_qwen_grpo_training.py` passed with
`28 passed`, and the full sim suite passed with `274 passed`.

The first RunPod action-likelihood smoke on the same two-sample slice was
manually stopped after it loaded Qwen onto the GPU but produced no sample log
for several minutes. The generated action-likelihood script now writes
`action_likelihood_progress.jsonl` before processor/model/adapter loading and
before/after every sample, so future remote smokes can identify whether a stall
is in model loading, adapter loading, input preparation, or forward scoring.

Follow-up RunPod action-likelihood checks used the pushed math-import fix
(`308b70a`) and the saved 8-step tool-balanced LoRA adapter:

```text
sim/scratch/open_vocab_nav_research_loop/qwen_grpo_action_likelihood_checks/
  stride5_24_lora8_toolbalanced_remote_mathfix_smoke1
  stride5_24_lora8_toolbalanced_remote_mathfix_24
```

The one-sample smoke completed cleanly and proved the diagnostic can load the
processor, Qwen model, PEFT adapter, dataset, and score a reference action. The
24-sample held-out slice also completed (`returncode=0`, `duration_s=745.751`,
`progress_count=58`, `tool_span_found_rate=1.0`), but the adapter did not show
a useful action-choice learning signal:

- Overall target mean logprob delta: `-0.008491996`; improved rate: `0.375`.
- Overall tool mean logprob delta: `-0.028123895`; improved rate: `0.291667`.
- By expected tool, target/tool deltas were
  `check_object_grounding=-0.024772845/-0.064817849`,
  `turn_by_angle=0.003706752/0.003371282`,
  `visual_servo_object=-0.005039766/-0.02620013`, and
  `stop=-0.00000081/-0.000000042`.

This is a go/no-go datapoint against scaling the current offline GRPO recipe
unchanged: it slightly improves `turn_by_angle`, but makes grounding and visual
servo action tokens less likely. The next training attempt should pivot to a
tiny overfit/SFT or action-token reward-shaping diagnostic before launching
larger GRPO runs.

The first tiny SFT pivot used the new `qwen_sft_training` module on RunPod:

```text
sim/scratch/open_vocab_nav_research_loop/qwen_tool_training/
  scan_20260614_130240_training_export
sim/scratch/open_vocab_nav_research_loop/qwen_sft_jobs/
  arrival_stop_tiny1
```

The source handoff came from
`open_vocab_nav_arrival_stop_bathroom_20260614_130240`, which produced
`accepted_count=6` SFT records and `rejected_count=12`. A one-sample LoRA SFT
job (`max_steps=10`, `learning_rate=2e-5`, `lora_r=8`) completed on RunPod with
`returncode=0`, `duration_s=342.523`, and an adapter saved under
`arrival_stop_tiny1/adapter`. The training log shows a real one-example
learning signal on a 9,397-token multimodal prompt: loss decreased from
`0.323851585` at step 1 to `0.226456374` at step 10.

This does not prove navigation improvement yet, but it does prove the simpler
teacher-forced path can change Qwen on a clean action target. Next verify the
SFT adapter with a teacher-forced likelihood or exact-completion check on the
same sample, then scale to the six accepted samples before returning to GRPO.

The follow-up SFT likelihood diagnostic adds an eval-only teacher-forced check
for SFT adapters:

```bash
python -m flatdisk_sim.qwen_sft_likelihood \
  --sft-training-job sim/scratch/open_vocab_nav_research_loop/qwen_sft_jobs/arrival_stop_tiny1 \
  --output-dir sim/scratch/open_vocab_nav_research_loop/qwen_sft_likelihood_checks/arrival_stop_tiny1_same_sample

python -m flatdisk_sim.qwen_sft_likelihood run \
  --job sim/scratch/open_vocab_nav_research_loop/qwen_sft_likelihood_checks/arrival_stop_tiny1_same_sample/qwen_sft_likelihood_job.json
```

It loads the base Qwen model plus a PEFT adapter, scores the exact assistant
message content used by SFT, compares `model.disable_adapter()` against the
adapter-enabled model under `torch.inference_mode()`, and writes both per-sample
JSONL and progress JSONL. Local tests cover raw-target preservation,
image-path precedence, adapter blockers, forbidden-token blockers, malformed
likelihood logs, metric aggregation, and direct `python -m` plan/run usage.

RunPod results:

- `arrival_stop_tiny1_same_sample`: one trained `visual_servo_object` sample,
  `returncode=0`, `duration_s=173.557`, full-target mean logprob delta
  `+0.112574629`, improved rate `1.0`, target logprob sum delta `+15.197574916`.
  The isolated action-tool token delta was approximately neutral/slightly
  negative (`-0.00005673`).
- `arrival_stop_six_sample_sft`: six accepted `visual_servo_object` SFT records,
  `max_steps=18`, `returncode=0`, `duration_s=487.045`, and loss decreased to
  `0.118048064`.
- `arrival_stop_six_sample_sft_trainset`: six-sample likelihood check over that
  adapter, `returncode=0`, `duration_s=295.687`, full-target mean logprob delta
  `+0.205490324`, improved rate `1.0`, min/max `+0.193296303/+0.21970717`.
  The isolated action-tool token delta stayed slightly negative on average
  (`-0.001529031`).

Interpretation: teacher-forced LoRA SFT gives a robust train-set learning signal
on the accepted visual-servo action records, unlike the current GRPO recipe.
However, because the tool-name token itself did not improve, the next diagnostic
should either add an action-token-focused loss/eval or run exact-completion
sampling to verify the adapter actually emits the desired tool call, not merely
the surrounding response structure.

The exact-completion SFT diagnostic now plans and runs deterministic
base-disabled versus adapter-enabled generations from an existing
`qwen_sft_training_job.json`:

```bash
python -m flatdisk_sim.qwen_sft_completion \
  --sft-training-job sim/scratch/open_vocab_nav_research_loop/qwen_sft_jobs/arrival_stop_six_sample_sft \
  --output-dir sim/scratch/open_vocab_nav_research_loop/qwen_sft_completion_checks/arrival_stop_six_sample_sft_trainset_action_recovered_140 \
  --max-samples 6 \
  --max-new-tokens 140

python -m flatdisk_sim.qwen_sft_completion run \
  --job sim/scratch/open_vocab_nav_research_loop/qwen_sft_completion_checks/arrival_stop_six_sample_sft_trainset_action_recovered_140
```

The generated script loads the same base model and PEFT adapter, generates once
with `model.disable_adapter()` and once with the adapter enabled, then parses
the emitted `action` object. The parser intentionally separates strict full JSON
parsing from tool-call recovery: if a completion begins with a balanced
`"action": {...}` object but is truncated later in audit/rationale text, the
tool call still counts while exact JSON/text matches remain false.

RunPod exact-completion result:

- `arrival_stop_six_sample_sft_trainset_action_recovered_140`: six train-set
  samples, `returncode=0`, all target tools were `visual_servo_object`.
  Base parsed an action in `6/6`, emitted `visual_servo_object` in `5/6`, and
  exactly matched the target action in `0/6`. The SFT adapter parsed an action
  in `6/6`, emitted `visual_servo_object` in `5/6`, and exactly matched the
  target action in `2/6`. The adapter improved the tool choice on one sample
  (`turn_by_angle` to `visual_servo_object`) and regressed one sample
  (`visual_servo_object` to `check_object_grounding`).
- Artifacts copied locally under
  `sim/scratch/open_vocab_nav_research_loop/qwen_sft_completion_checks/arrival_stop_six_sample_sft_trainset_action_recovered_140/`:
  `qwen_sft_completion_result.json`,
  `qwen_sft_completion_samples.jsonl`, and
  `qwen_sft_completion_progress.jsonl`.

Interpretation: the exact completion check confirms the SFT adapter can change
Qwen's generated action, and it improves exact action arguments on this tiny
train set. It does not yet improve top-level tool choice over base Qwen
(`5/6` versus `5/6`). The next training step should use an action-focused target
or loss, and the next eval should use a held-out/diverse slice instead of only
six accepted visual-servo records.

Resume status for a future continuation:

- This is no longer a verification-only blocker. Runpod auth works when
  `RUNPOD_API_KEY` is loaded from the repo root `.env`, the A40 pod can run the
  training stack, and one-step, four-step, eight-step, and exact-bonus LoRA
  adapters were saved successfully.
- No user action is currently required to start the next training attempt.
- Use the pushed `codex/open-vocab-nav-research-loop` ref for a clean checkout;
  the original `/workspace/flat-disk-robot-code` directory on the pod is not a
  git repo.
- The next useful experiment is action-choice learning signal and evaluation
  scaling. The JSON-format issue is no longer the main blocker, generic partial
  reward is in place, and reference-tool balancing improves early same-tool
  behavior; the 8-step diagnostic still shows weak exact-action selection and
  persistent `turn_by_angle` fallback; the exact-action bonus ablation did not
  help on the short comparison; held-out completion-eval planning and a first
  base-versus-adapter eval now exist. Useful next directions are adding an
  adapter-effect sanity check, collecting more diverse grouped rollouts with
  stop/arrival and ambiguous grounding examples, auditing whether reference
  actions are too noisy or too correlated with turns, and then running a
  stronger balanced comparison. Keep `--max-completion-length` explicit so smoke
  and short training jobs cannot spend most of their time generating long
  completions.

The fixes needed to make the smoke complete were:

- Install the training stack in a venv with `--system-site-packages` so it can
  use the image's global Torch `2.9.1+cu128`.
- Install `torchvision==0.24.1 --no-deps` to match that Torch build; installing
  unconstrained latest `torchvision` tries to pull a second Torch/CUDA stack.
- Keep GRPO prompts conversational. Do not pre-apply the Qwen chat template.
- Convert structured message content to text strings before `Dataset.from_list`
  and keep loaded PIL images in the separate `images` column so TRL can insert
  image placeholders itself.
- Use PEFT LoRA adapters; full-model Adam OOMs on the A40 at the first optimizer
  step.
- Cap `max_completion_length` for smoke jobs; uncapped generation made a tiny
  run take too long.
- When using the persistent pod venv from a fresh checkout, set
  `PYTHONPATH=sim/src` before `flatdisk-sim-run-qwen-grpo-training` so the entry
  point imports the checked-out source instead of an older editable install.

With `--start-qwen-server`, the generated worker claims the Warmhub task, starts
`Qwen/Qwen3-VL-8B-Instruct` through vLLM, waits for `/v1/models`, and then runs
the planned command with `--no-claim`. The Qwen server log is attached as task
evidence.

To fan out several planned trial-slice tasks, use the dispatcher. It queries
Warmhub `AgentTask` records, filters locally, and emits one Runpod pod command
per selected task:

```bash
uv run --project sim flatdisk-sim-runpod-dispatch \
  --name-prefix qwen-strategy-runpod-linux-v4-run- \
  --tag trial-slice \
  --max-workers 2 \
  --start-qwen-server \
  --qwen-vllm-extra-args "--max-model-len 8192" \
  --dispatch-manifest sim/scratch/open_vocab_nav_research_loop/runpod_dispatch_manifest.json \
  --terminate-after 4h \
  --evidence-artifact /workspace/outputs/open_vocab_nav_research_loop
```

Add `--launch` only after reviewing the dry-run output. The dispatcher refuses
real launches from a dirty worktree unless `--allow-dirty` is supplied.
The dispatch manifest is a redacted, durable review artifact; attach it to
`SubAgentResult.evidenceArtifacts` or cite it from an `AgentNote` so future
agents can reconstruct which Warmhub tasks were selected, skipped, and launched
from which git ref.
It also includes bounded running and blocked `AgentTask` samples so reviewers
can see active queue pressure before adding more Runpod workers, and flags
running tasks whose `updatedAt` looks stale.
The dispatcher reads a larger Warmhub task page before local filtering so older
matching planned tasks are not missed in a busy queue.
By default it skips tasks whose `notes.prerequisites` are not complete; pass
`--ignore-prerequisites` only for deliberate recovery/debug work.

## Generality Rules

- No color-threshold object detector in the model-facing path.
- No hard-coded object-family policy should count as a research-loop baseline.
- THOR target labels and object metadata are evaluator-only.
- Florence/GroundingDINO-style phrase grounding is a tool result, not proof of
  semantic goal completion.
- Research configs default to `strict_model_based=true`, which rejects
  non-model runners and THOR-derived semantic topomap term routing before a run
  starts. Set it false only for explicit smoke/debug experiments.
- `noHardcodedLabelsOrColors` is derived, not asserted: strict results require a
  model-based runner (`qwen` or `codex`) and no static object/color examples in
  the model-facing prompt context.
- Topomap memory hints are navigation memory, not ground-truth completion.
- Scripted or fast hard-coded runners may be used as smoke tests, but not as
  evidence of a general model-based solution.

## Next Integrations

- Run `qwen-strategy-runpod-linux-v4-preflight` with `START_QWEN_SERVER=1` on
  Runpod, then fan out v4 trial slices after preflight completes.
- Use `flatdisk-sim-prepare-qwen-tool-training` before PPO/GRPO to join
  `training_export/policy_dataset_v1/policy_samples.jsonl` with
  `policy_dataset_v1/evaluator_labels.jsonl`, resolve image paths, emit
  Qwen-compatible multimodal SFT JSONL plus `qwen_action_preferences.jsonl` for
  guard-replaced actor actions plus `qwen_dpo_messages.jsonl` for explicit
  prompt/chosen/rejected preference training, and write an audit report covering
  reward filters, missing images, actor-vs-executed action targets, and
  privileged-token leakage.
- Run `flatdisk-sim-nav-training-readiness` over both the `training_export/`
  tree and any `qwen_tool_training/` output before choosing a training method.
  The readiness assertion now reports raw policy samples, accepted Qwen SFT
  records, Qwen guard-replacement preferences, DPO handoff records, missing Qwen
  images, and privileged-token scans separately.
- Use `python -m flatdisk_sim.qwen_sft_training --input <qwen_tool_training>`
  to plan a tiny teacher-forced Qwen LoRA SFT job from
  `qwen_sft_messages.jsonl`; use
  `python -m flatdisk_sim.qwen_sft_training run --job <qwen_sft_training_job.json>`
  on a GPU worker. This is the preferred next diagnostic after the negative
  action-likelihood result: first prove one to a few clean action JSON targets
  can be overfit before launching larger GRPO runs.
- Use `flatdisk-sim-plan-qwen-dpo-training` after readiness to validate
  `qwen_dpo_messages.jsonl`, generate a `qwen_dpo_training_job.json`, and emit a
  Runpod-oriented TRL script without importing GPU training dependencies in the
  local sim test environment.
- Use `flatdisk-sim-run-qwen-dpo-training --job <qwen_dpo_training_job.json>`
  only on a training-capable worker. The wrapper records
  `qwen_dpo_training_result.json`, checks required packages via import discovery
  before launch, and leaves the heavy Transformers/TRL imports inside the
  generated script.
- Use `flatdisk-sim-prepare-qwen-grpo-training` after readiness to convert
  `rollout_groups.jsonl` into `qwen_grpo_rollout_groups.jsonl`, with per-step
  Qwen prompt/assistant targets and trajectory rewards held in a separate
  evaluator channel for later GRPO/PPO training. Pass repeated `--input`
  values to merge split one-rollout Runpod exports into a comparable group.
- Use `flatdisk-sim-plan-qwen-grpo-training` on a ready
  `qwen_grpo_training_manifest.json` to write the offline replay GRPO job
  manifest, prompt/image JSONL, and generated TRL script. Run
  `flatdisk-sim-run-qwen-grpo-training --job <qwen_grpo_training_job.json>`
  only on a training-capable worker; use `--dry-run --skip-dependency-check`
  for queue/path validation.
- Dispatch preference-training GPU workers with `--stage preference-training`
  and `--tag gpu-training-worker --no-start-thor-xorg`; do not pass
  `--start-qwen-server` because DPO/GRPO training loads the model directly.
- Use `training_export/policy_review_traces.jsonl` first for failure triage and
  parallel-agent handoff; it records tool calls and contact-sheet paths without
  hidden target distances or THOR object metadata.
- Use `flatdisk-sim-analyze-nav-failures` to turn policy-review traces into
  WarmHub AgentNotes with trace-grounded prompt/tool recommendations.
- Use `rollout_groups.jsonl` / `trajectory_preferences.jsonl` for trajectory
  ranking only after the materializer can prove policy inputs remain clean and
  there are enough successful and failed runs for a meaningful comparison.
- Add model variants beyond Qwen3-VL 8B: small open VLMs for planning, separate
  critics, and alternate open-vocabulary grounding tools.
- Convert successful and failed `NavEvalRun` records into PPO/GRPO training data
  only after grouped rollouts, reward normalization/advantages, and on-policy
  rollout collection are explicit rather than inferred from offline labels.
