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

Resume status for a future continuation:

- This is no longer a verification-only blocker. Runpod auth works when
  `RUNPOD_API_KEY` is loaded from the repo root `.env`, the A40 pod can run the
  training stack, and one-step, four-step, and eight-step LoRA adapters were saved
  successfully.
- No user action is currently required to start the next training attempt.
- Use the pushed `codex/open-vocab-nav-research-loop` ref for a clean checkout;
  the original `/workspace/flat-disk-robot-code` directory on the pod is not a
  git repo.
- The next useful experiment is action-choice learning signal and evaluation
  scaling. The JSON-format issue is no longer the main blocker, generic partial
  reward is in place, and reference-tool balancing improves early same-tool
  behavior; the 8-step diagnostic still shows weak exact-action selection and
  persistent `turn_by_angle` fallback. Useful next directions are adding a
  generic exact-action bonus ablation for zero-reward tools, collecting more
  diverse grouped rollouts with stop/arrival and ambiguous grounding examples,
  and then running a 24-step balanced comparison or held-out completion eval.
  Keep `--max-completion-length` explicit so smoke and short training jobs
  cannot spend most of their time generating long completions.

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
