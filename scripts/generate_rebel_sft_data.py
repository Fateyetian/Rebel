#!/usr/bin/env python3
"""
ReBel SFT data generation — two-stage pipeline.

Stage 1 (annotate): call teacher LLM per step → save annotated trajectories
Stage 2 (convert):  read annotated trajectories → write SFT JSONL pairs

Usage
-----
# Stage 1: annotate
python3 scripts/generate_rebel_sft_data.py \\
    --stage annotate \\
    --env alfworld \\
    --input  data/alfworld_rebel/rebel_coldstart.json \\
    --annotated data/alfworld_sft_annotated/annotated_trajs.jsonl \\
    --api_base https://api.openai.com/v1 \\
    --api_key  $TEACHER_API_KEY \\
    --model    gpt-4o \\
    --workers  8

# Stage 2: convert annotated trajs → SFT pairs
python3 scripts/generate_rebel_sft_data.py \\
    --stage convert \\
    --env alfworld \\
    --annotated data/alfworld_sft_annotated/annotated_trajs.jsonl \\
    --output    data/alfworld_sft/train.jsonl

# Both stages in sequence (default)
python3 scripts/generate_rebel_sft_data.py \\
    --stage both \\
    --env alfworld \\
    --input     data/alfworld_rebel/rebel_coldstart.json \\
    --annotated data/alfworld_sft_annotated/annotated_trajs.jsonl \\
    --output    data/alfworld_sft/train.jsonl \\
    --api_base  https://api.openai.com/v1 \\
    --api_key   $TEACHER_API_KEY \\
    --model     gpt-4o \\
    --workers   8

Annotated trajectory format (one per line in annotated_trajs.jsonl):
  {
    "id": "ep_0",
    "task": "...",
    "task_type": "pick_and_place",
    "steps": [
      {"idx": 0, "obs": "...", "admissible": "...",
       "ground_truth_action": "...", "next_obs": "...",
       "llm_response": "<belief>...</belief><think>...</think><action>...</action>",
       "belief_json": "..."}
    ]
  }

SFT pair format (one per line in train.jsonl):
  {"prompt": "<RL-format prompt identical to training>", "response": "<belief><think><action>"}
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


# ─────────────────────────────────────────────────────────────────────────────
# Template imports (keeps SFT prompt = RL prompt)
# ─────────────────────────────────────────────────────────────────────────────

def _alf_templates():
    from agent_system.environments.env_package.alfworld.alfworld_rebel_prompt import (
        ALFWORLD_TEMPLATE_REBEL,
        ALFWORLD_TEMPLATE_NO_HIS_REBEL,
        ALFWORLD_REBEL_TAGGING_TEMPLATE,
        ALFWORLD_TEMPLATE_REBEL_SFT,
        ALFWORLD_TEMPLATE_NO_HIS_REBEL_SFT,
    )
    # Returns: (rl_hist, rl_nohist, tagging, sft_hist, sft_nohist)
    return (
        ALFWORLD_TEMPLATE_REBEL,
        ALFWORLD_TEMPLATE_NO_HIS_REBEL,
        ALFWORLD_REBEL_TAGGING_TEMPLATE,
        ALFWORLD_TEMPLATE_REBEL_SFT,
        ALFWORLD_TEMPLATE_NO_HIS_REBEL_SFT,
    )


def _ws_templates():
    from agent_system.environments.prompts.webshop_rebel_prompts import (
        WEBSHOP_REBEL_TEMPLATE_RL,
        WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL,
        WEBSHOP_REBEL_TAGGING_TEMPLATE,
        WEBSHOP_REBEL_TEMPLATE_SFT,
        WEBSHOP_REBEL_TEMPLATE_NO_HIS_SFT,
    )
    # Returns: (rl_hist, rl_nohist, tagging, sft_hist, sft_nohist)
    return (
        WEBSHOP_REBEL_TEMPLATE_RL,
        WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL,
        WEBSHOP_REBEL_TAGGING_TEMPLATE,
        WEBSHOP_REBEL_TEMPLATE_SFT,
        WEBSHOP_REBEL_TEMPLATE_NO_HIS_SFT,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape literal braces so str.format() won't misparse dynamic content."""
    return s.replace('{', '{{').replace('}', '}}')


def call_llm(client: OpenAI, model: str, prompt: str,
             retries: int = 3, temperature: float = 0.2) -> Optional[str]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [LLM error attempt {attempt+1}/{retries}]: {e}  (retry in {wait}s)")
            time.sleep(wait)
    return None


def parse_belief_json(text: str) -> Optional[str]:
    m = re.search(r'<belief>(.*?)</belief>', text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    blob = m.group(1).strip()
    try:
        json.loads(blob)
        return blob
    except json.JSONDecodeError:
        try:
            fixed = blob.replace("'", '"')
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            return None


def _load_done_ep_ids(annotated_path: str) -> set:
    """Return set of already-annotated episode IDs for resume."""
    done = set()
    if os.path.exists(annotated_path):
        with open(annotated_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)['id'])
                    except Exception:
                        pass
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Admissible-action helpers
# ─────────────────────────────────────────────────────────────────────────────

def derive_alf_admissible(obs: str, action: str) -> str:
    actions = {action}
    obs_lower = obs.lower()
    objects = re.findall(r'\b([a-z]+ \d+)\b', obs_lower)
    receps = [o for o in objects if any(
        r in o for r in ('cabinet', 'drawer', 'shelf', 'table', 'counter', 'bed',
                         'microwave', 'fridge', 'sink', 'bathtub', 'toilet',
                         'sofa', 'desk', 'sidetable', 'ottoman', 'armchair',
                         'diningtable', 'garbagecan'))]
    for recep in receps[:8]:
        actions.add(f'go to {recep}')
    inv_m = re.search(r'you are holding (?:a |an )?([\w\s]+\d+)', obs_lower)
    if inv_m:
        held = inv_m.group(1).strip()
        for recep in receps[:4]:
            actions.add(f'put {held} in/on {recep}')
        for appl in ('microwave 1', 'fridge 1', 'sinkbasin 1'):
            if appl in obs_lower:
                actions.add(f'heat {held} with {appl}')
                actions.add(f'cool {held} with {appl}')
                actions.add(f'clean {held} with {appl}')
    return "\n ".join(f"'{a}'" for a in sorted(actions))


def derive_ws_admissible(action: str) -> str:
    actions = {action, 'Back to Search', 'search[<your query>]'}
    return "\n".join(f"'{a}'," for a in sorted(actions))


def task_type_from_description(task_desc: str) -> str:
    td = task_desc.lower()
    if 'heat' in td:  return 'pick_heat_then_place_in_recep'
    if 'cool' in td:  return 'pick_cool_then_place_in_recep'
    if 'clean' in td: return 'pick_clean_then_place_in_recep'
    if 'light' in td: return 'look_at_obj_in_light'
    if 'two' in td:   return 'pick_two_obj_and_place'
    return 'pick_and_place'


# ─────────────────────────────────────────────────────────────────────────────
# Phase vocabulary auto-correction (old → new)
# ─────────────────────────────────────────────────────────────────────────────

_ALF_PHASE_MAP = {
    'search': 'find',
    'pick':   'pickup',
    'pick_up': 'pickup',
}
_ALF_VALID_PHASES = {'find', 'navigate', 'pickup', 'transform', 'place', 'done'}


def _fix_alf_phase(belief_str: str) -> str:
    """Map old phase vocabulary to new in a parsed belief JSON string."""
    try:
        b = json.loads(belief_str)
        phase = b.get('task', {}).get('phase', '')
        corrected = _ALF_PHASE_MAP.get(phase, phase)
        if corrected not in _ALF_VALID_PHASES:
            corrected = 'find'  # safe fallback
        if corrected != phase:
            b.setdefault('task', {})['phase'] = corrected
            return json.dumps(b, ensure_ascii=False)
    except Exception:
        pass
    return belief_str


def _fix_alf_belief_post(belief_str: str, gt_action: str, next_obs: str) -> str:
    """
    Post-processing fixes applied after LLM annotation:

    Fix A — Deterministic prediction: overwrite the LLM's free-form prediction
    with a ground-truth-grounded one built from next_obs.  This eliminates the
    'If I go to X' / 'If I execute X' format inconsistency and ensures the
    prediction always references the correct action.

    Fix B — Navigate phase: if phase is 'find' but the target object's location
    is already known (not 'unknown' / 'in_hand') in belief.state.objects, the
    agent should have transitioned to 'navigate'.  Correct it deterministically.

    Fix C — In-hand after pick-up: if the current gt_action is a 'take' action
    and next_obs confirms a successful pick-up ("you pick up"), mark the object
    as 'in_hand' in objects and advance phase to 'pickup' (will become the
    previous belief for the very next step).
    """
    try:
        b = json.loads(belief_str)
    except Exception:
        return belief_str

    state = b.setdefault('state', {})
    task  = b.setdefault('task', {})
    objs  = state.setdefault('objects', {})

    # Fix A: deterministic prediction
    b['prediction'] = build_prediction(gt_action, next_obs)

    # Fix B: find → navigate when target location already confirmed
    if task.get('phase') == 'find':
        target = task.get('target', '')
        target_base = target.split()[0].lower() if target else ''
        for obj_id, loc in objs.items():
            if loc not in ('unknown', 'in_hand', '') and target_base and target_base in obj_id.lower():
                task['phase'] = 'navigate'
                break

    # Fix C: take action + successful next_obs → mark in_hand, advance phase
    if gt_action.lower().startswith('take ') and 'you pick up' in next_obs.lower():
        m = re.match(r'take ([\w\s]+\d+) from', gt_action, re.IGNORECASE)
        if m:
            obj_id = m.group(1).strip()
            objs[obj_id] = 'in_hand'
            if task.get('phase') in ('find', 'navigate', 'pickup'):
                task['phase'] = 'pickup'

    return json.dumps(b, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Belief merge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_alf_belief(cum: Dict, new: Dict):
    s = new.get('state', {}) or {}
    t = new.get('task', {}) or {}
    cum['state']['objects'].update(s.get('objects', {}) or {})
    cum['state']['states'].update(s.get('states', {}) or {})
    existing = set(cum['state']['visited'])
    for v in (s.get('visited', []) or []):
        if isinstance(v, str) and v and v not in existing:
            cum['state']['visited'].append(v)
            existing.add(v)
    # unvisited_candidates: accumulate new; remove those now visited
    visited_set = set(cum['state']['visited'])
    existing_uv = set(cum['state'].get('unvisited_candidates', []))
    for loc in (s.get('unvisited_candidates', []) or []):
        if isinstance(loc, str) and loc and loc not in visited_set and loc not in existing_uv:
            cum['state'].setdefault('unvisited_candidates', []).append(loc)
            existing_uv.add(loc)
    cum['state']['unvisited_candidates'] = [
        u for u in cum['state'].get('unvisited_candidates', []) if u not in visited_set
    ]
    for key in ('phase', 'target', 'subgoal'):
        val = t.get(key)
        if val and isinstance(val, str):
            cum['task'][key] = val


def _merge_ws_belief(cum: Dict, new: Dict):
    s = new.get('state', {}) or {}
    t = new.get('task', {}) or {}
    cum['state']['target'].update(s.get('target', {}) or {})
    cum['state']['confirmed'].update(s.get('confirmed', {}) or {})
    confirmed_keys = set(cum['state']['confirmed'])
    cum['state']['unconfirmed'] = [
        u for u in cum['state']['unconfirmed'] if u not in confirmed_keys
    ]
    existing = set(cum['state']['unconfirmed'])
    for attr in (s.get('unconfirmed', []) or []):
        if isinstance(attr, str) and attr and attr not in existing and attr not in confirmed_keys:
            cum['state']['unconfirmed'].append(attr)
            existing.add(attr)
    pid = s.get('product_id')
    if pid and str(pid).strip() not in ('', 'null', 'None'):
        cum['state']['product_id'] = str(pid).strip()
    for key in ('phase', 'target', 'subgoal'):
        val = t.get(key)
        if val is not None and str(val).strip():
            cum['task'][key] = val


def _fix_ws_belief_post(belief_str: str, gt_action: str, next_obs: str) -> str:
    """
    Post-processing fixes for WebShop belief annotation:

    Fix A — Deterministic prediction: overwrite with ground-truth-grounded prediction
    built from next_obs, eliminating empty / incorrectly formatted LLM predictions.

    Fix B — click[ASIN] phase transition: clicking a product ASIN moves from
    searching/browsing to 'viewing'; set product_id to the ASIN.

    Fix C — click[Buy Now] phase transition: Buy Now must set phase to 'buying'.

    Fix D — confirmed monotonicity: confirmed entries from previous belief must be
    preserved (applied via _merge_ws_belief, but catch regressions here).

    Fix E — inferred text-attribute confirmation: WebShop product detail pages do NOT
    expose feature text (material, sleeve, care, fit, closure, etc.) in the obs — only
    size/color options and section-header links are shown. When the agent transitions to
    'selecting' or 'buying', it has implicitly judged that the product matches the task
    requirements. For every target attribute that is still absent from confirmed, inject
    it with value "inferred: <target_value>" so downstream reward computation can treat
    these as soft-confirmed rather than unknown. This avoids the "partial-match as
    full-match" problem while accurately reflecting that the evidence is search-intent
    based, not directly observed.
    """
    try:
        b = json.loads(belief_str)
    except Exception:
        return belief_str

    state = b.setdefault('state', {})
    task  = b.setdefault('task', {})

    # Fix A: deterministic prediction
    b['prediction'] = build_prediction(gt_action, next_obs)

    # Fix B: click[ASIN] → viewing
    asin_m = re.match(r'click\[([A-Z0-9]{10})\]', gt_action)
    if asin_m:
        asin = asin_m.group(1)
        state['product_id'] = asin
        if task.get('phase') in ('searching', 'browsing', None, ''):
            task['phase'] = 'viewing'

    # Fix C: click[Buy Now] → buying
    if gt_action == 'click[Buy Now]':
        task['phase'] = 'buying'

    # Fix E: infer unconfirmed text attributes from target when selecting/buying
    # (product features are not in obs; expert selection implies implicit match)
    _OBS_ONLY_KEYS = {'search_results', 'product_id', 'product_name',
                      'price', 'price_range', 'price_within_budget'}
    if task.get('phase') in ('selecting', 'buying'):
        target    = state.get('target', {})
        confirmed = state.setdefault('confirmed', {})
        unconfirmed = state.get('unconfirmed', [])
        newly_confirmed = []
        for attr, val in target.items():
            if attr in _OBS_ONLY_KEYS:
                continue
            # skip if already confirmed directly (any key containing attr name)
            already = any(attr.replace(' ','_') in k or k.startswith(attr.split()[0])
                          for k in confirmed)
            if not already and val:
                confirmed[f'{attr}'] = f'inferred: {val}'
                newly_confirmed.append(attr)
        # remove inferred attrs from unconfirmed list
        state['unconfirmed'] = [u for u in unconfirmed
                                 if u not in newly_confirmed and
                                 not any(u.startswith(nc) for nc in newly_confirmed)]

    return json.dumps(b, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic prediction builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction(action: str, next_obs: str, max_len: int = 100) -> str:
    """
    Construct a concise, guaranteed-correct prediction from ground-truth next_obs.
    Format matches the prompt template: "If I execute [action], I expect [obs]."

    Strategy:
    - Action-result sentences ("You pick up X", "You put X on Y") are short → keep as-is
    - Navigation sentences ("On the X, you see A, B, C, ...") can be very long →
      truncate to first 2 objects to keep prediction learnable during RL inference
    """
    summary = next_obs.strip()

    # Take first sentence
    end = summary.find('.')
    if 0 < end < 400:
        summary = summary[:end + 1]

    # If already short enough, done
    if len(summary) <= max_len:
        return f"If I execute {action}, I expect {summary}"

    # Long sentence (object listing): keep up to second comma after "you see"
    see_idx = summary.lower().find('you see')
    if see_idx >= 0:
        commas = [i for i, c in enumerate(summary) if c == ',' and i > see_idx]
        if len(commas) >= 2:
            summary = summary[:commas[1]] + '...'
        elif commas:
            summary = summary[:commas[0]] + '...'
        else:
            summary = summary[:max_len] + '...'
    else:
        summary = summary[:max_len] + '...'

    return f"If I execute {action}, I expect {summary}"


def inject_prediction(llm_response: str, action: str, next_obs: str) -> str:
    """Replace the prediction field in the LLM belief JSON with a deterministic one."""
    pred = build_prediction(action, next_obs)
    belief_m = re.search(r'<belief>(.*?)</belief>', llm_response, re.DOTALL | re.IGNORECASE)
    if not belief_m:
        return llm_response
    try:
        belief_dict = json.loads(belief_m.group(1).strip())
        belief_dict['prediction'] = pred
        new_belief = json.dumps(belief_dict, indent=2, ensure_ascii=False)
        return llm_response[:belief_m.start(1)] + '\n' + new_belief + '\n' + llm_response[belief_m.end(1):]
    except Exception:
        # Fallback: regex replace prediction value
        new_resp = re.sub(
            r'"prediction"\s*:\s*"[^"]*"',
            f'"prediction": {json.dumps(pred)}',
            llm_response,
        )
        return new_resp


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Annotate ALFWorld trajectories
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_alf_episode(ep_idx: int, ep: Dict, client: OpenAI, model: str,
                           tmpl_tag: str) -> Optional[Dict]:
    """Annotate one ALFWorld episode sequentially (belief chain dependency)."""
    task_desc = ep['task']
    steps     = ep.get('data', ep.get('traj', []))
    task_type = task_type_from_description(task_desc)
    n_steps   = len(steps)

    cum_belief: Dict = {
        'state': {'objects': {}, 'states': {}, 'visited': [], 'unvisited_candidates': []},
        'task': {'phase': 'find', 'target': '', 'subgoal': 'Find the target object'},
    }
    history: List[Tuple[str, str]] = []
    annotated_steps = []

    for step_idx, step in enumerate(steps):
        obs  = step['obs']
        resp = step['response']

        # Extract ground-truth action from old response
        action_m = re.search(r'<action>(.*?)</action>', resp, re.DOTALL | re.IGNORECASE)
        if action_m:
            gt_action = action_m.group(1).strip()
        else:
            gt_action = next((l.strip() for l in reversed(resp.split('\n')) if l.strip()), 'look')

        next_obs   = steps[step_idx + 1]['obs'] if step_idx + 1 < n_steps else obs
        admissible = derive_alf_admissible(obs, gt_action)

        # Build action history from recent steps (window=2, matches convert stage)
        recent = history[-2:]
        action_history_lines = [
            f"[Step {len(history) - len(recent) + j + 1}: Observation: '{h_obs[:80]}', Action: '{h_act}']"
            for j, (h_obs, h_act) in enumerate(recent)
        ]
        action_history_str = '\n'.join(action_history_lines) if action_history_lines else '(start of episode)'

        # Extract expert_think from original response (<think> or <reasoning> tag)
        think_m = re.search(r'<think>(.*?)</think>', resp, re.DOTALL | re.IGNORECASE)
        if not think_m:
            think_m = re.search(r'<reasoning>(.*?)</reasoning>', resp, re.DOTALL | re.IGNORECASE)
        expert_think = think_m.group(1).strip() if think_m else ''

        current_step = step_idx + 1
        step_count   = step_idx

        tag_prompt = tmpl_tag.format(
            task_description=_esc(task_desc),
            step_count=step_count,
            history_length=len(recent),
            action_history=_esc(action_history_str),
            current_step=current_step,
            current_observation=_esc(obs),
            previous_belief=_esc(json.dumps(cum_belief, ensure_ascii=False)),
            expert_think=_esc(expert_think),
            expert_action=_esc(gt_action),
        )

        llm_resp = call_llm(client, model, tag_prompt)
        if not llm_resp:
            print(f"    ep{ep_idx+1}/step{step_idx}: LLM failed, using empty belief")
            belief_json_str = json.dumps(cum_belief)
            llm_resp = f"<belief>{belief_json_str}</belief>\n<think>N/A</think>\n<action>{gt_action}</action>"
        else:
            belief_json_str = parse_belief_json(llm_resp)
            if belief_json_str:
                belief_json_str = _fix_alf_phase(belief_json_str)
            else:
                belief_json_str = json.dumps(cum_belief)

        # Post-processing: deterministic prediction + navigate phase + in_hand
        belief_json_str = _fix_alf_belief_post(belief_json_str, gt_action, next_obs)

        annotated_steps.append({
            'idx': step_idx,
            'obs': obs,
            'admissible': admissible,
            'ground_truth_action': gt_action,
            'next_obs': next_obs,
            'llm_response': llm_resp.strip(),
            'belief_json': belief_json_str,
        })

        # Update cumulative belief with LLM output for next step
        try:
            _merge_alf_belief(cum_belief, json.loads(belief_json_str))
        except Exception:
            pass

        history.append((obs, gt_action))

    return {
        'id': f'ep_{ep_idx}',
        'task': task_desc,
        'task_type': task_type,
        'steps': annotated_steps,
    }


def annotate_alfworld(input_path: str, annotated_path: str,
                      client: OpenAI, model: str,
                      max_trajectories: int, workers: int):
    _, _, tmpl_tag, _, _ = _alf_templates()

    with open(input_path) as f:
        episodes = json.load(f)
    if max_trajectories > 0:
        episodes = episodes[:max_trajectories]

    done_ids = _load_done_ep_ids(annotated_path)
    os.makedirs(os.path.dirname(os.path.abspath(annotated_path)), exist_ok=True)
    out_f    = open(annotated_path, 'a')
    out_lock = Lock()

    print(f"Episodes: {len(episodes)}  |  Already done: {len(done_ids)}  |  Workers: {workers}")

    def process_ep(args):
        idx, ep = args
        ep_id = f'ep_{idx}'
        if ep_id in done_ids:
            return None
        result = _annotate_alf_episode(idx, ep, client, model, tmpl_tag)
        print(f"  [{idx+1}/{len(episodes)}] task={ep['task'][:50]}  steps={len(result['steps'])}")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_ep, (i, ep)): i for i, ep in enumerate(episodes)}
        done_count = 0
        written = 0
        for fut in as_completed(futures):
            done_count += 1
            result = fut.result()
            if result is not None:
                with out_lock:
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    out_f.flush()
                    written += 1

    out_f.close()
    total = len(done_ids) + written
    print(f"\nStage 1 done. {total} trajectories in {annotated_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Convert annotated ALFWorld → SFT JSONL
# ─────────────────────────────────────────────────────────────────────────────

def convert_alfworld(annotated_path: str, output_path: str):
    _, _, _, tmpl_sft_hist, tmpl_sft_nohist = _alf_templates()

    records = []
    with open(annotated_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pairs = []

    for rec in records:
        task_desc = rec['task']
        task_type = rec.get('task_type', task_type_from_description(task_desc))
        steps     = rec['steps']

        # Rebuild cumulative belief from annotated outputs
        cum_belief: Dict = {
            'state': {'objects': {}, 'states': {}, 'visited': [], 'unvisited_candidates': []},
            'task': {'phase': 'find', 'target': '', 'subgoal': 'Find the target object'},
        }
        history: List[Tuple[str, str]] = []

        for step in steps:
            step_idx   = step['idx']
            obs        = step['obs']
            admissible = step['admissible']
            gt_action  = step['ground_truth_action']
            llm_resp   = step['llm_response']
            belief_str = step['belief_json']

            action_history = ''
            for j, (h_obs, h_act) in enumerate(history[-2:]):
                sn = len(history) - len(history[-2:]) + j + 1
                action_history += f"\n[Step {sn}: Observation: '{h_obs[:80]}', Action: '{h_act}']"

            if step_idx == 0:
                sft_prompt = tmpl_sft_nohist.format(
                    task_description=_esc(task_desc),
                    current_observation=_esc(obs),
                )
            else:
                sft_prompt = tmpl_sft_hist.format(
                    task_description=_esc(task_desc),
                    step_count=step_idx,
                    history_length=len(history[-2:]),
                    action_history=_esc(action_history.strip()),
                    current_step=step_idx + 1,
                    current_observation=_esc(obs),
                    previous_belief=_esc(json.dumps(cum_belief, indent=2, ensure_ascii=False)),
                )

            # Deterministically overwrite prediction with ground-truth next_obs
            next_obs = step.get('next_obs', '')
            if next_obs:
                llm_resp = inject_prediction(llm_resp, gt_action, next_obs)

            # Tagging template no longer outputs <action>; append ground-truth action
            if '<action>' not in llm_resp:
                llm_resp = llm_resp.strip() + f'\n\n<action>\n{gt_action}\n</action>'

            pairs.append({'prompt': sft_prompt, 'response': llm_resp})

            # Advance cumulative belief using LLM-generated belief
            try:
                _merge_alf_belief(cum_belief, json.loads(belief_str))
            except Exception:
                pass
            history.append((obs, gt_action))

    with open(output_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"Stage 2 done. {len(pairs)} SFT pairs written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Annotate WebShop trajectories
# ─────────────────────────────────────────────────────────────────────────────

def _extract_webshop_steps(ep: Dict) -> List[Dict]:
    if 'conversations' in ep:
        convs = ep['conversations']
        human_turns = [c['value'] for c in convs if c.get('from') == 'human']
        gpt_turns   = [c['value'] for c in convs if c.get('from') == 'gpt']
        steps = []
        for i, (h, g) in enumerate(zip(human_turns, gpt_turns)):
            am = re.search(r'<action>(.*?)</action>', g, re.DOTALL | re.IGNORECASE)
            if not am:
                am = re.search(r'(?:^|\n)((?:search|click)\[.*?\])', g, re.MULTILINE)
            action = am.group(1).strip() if am else ''
            next_obs = human_turns[i + 1] if i + 1 < len(human_turns) else h
            steps.append({'obs': h, 'action': action, 'next_obs': next_obs})
        return steps
    return ep.get('steps', [])


def _extract_ws_task(ep: Dict) -> str:
    if 'task' in ep:
        return ep['task']
    for c in ep.get('conversations', []):
        if c.get('from') == 'human':
            m = re.search(r'Instruction:\s*(.+?)(?:\n|$)', c['value'])
            if m:
                return m.group(1).strip()
    return ''


def _annotate_ws_episode(ep_idx: int, ep: Dict, client: OpenAI, model: str,
                          tmpl_tag: str) -> Optional[Dict]:
    task_desc = _extract_ws_task(ep)
    steps     = _extract_webshop_steps(ep)
    if not task_desc or not steps:
        return None

    cum_belief: Dict = {
        'state': {'target': {}, 'confirmed': {}, 'unconfirmed': [], 'product_id': None},
        'task': {'phase': 'searching', 'target': None, 'subgoal': 'Search for target product'},
    }
    history: List[str] = []
    annotated_steps = []

    for step_idx, step in enumerate(steps):
        obs      = step['obs']
        action   = step['action']
        next_obs = step['next_obs']
        if not action:
            history.append(action)
            continue

        admissible = derive_ws_admissible(action)
        ctx = '\n'.join(f"  Step {i+1}: {a}" for i, a in enumerate(history[-4:])) or '(start)'

        # Extract expert_think from any existing response annotation
        think_m = re.search(r'<think>(.*?)</think>', step.get('response', ''), re.DOTALL | re.IGNORECASE)
        if not think_m:
            think_m = re.search(r'<reasoning>(.*?)</reasoning>', step.get('response', ''), re.DOTALL | re.IGNORECASE)
        expert_think = think_m.group(1).strip() if think_m else ''

        current_step = step_idx + 1
        step_count   = step_idx

        tag_prompt = tmpl_tag.format(
            task_description=_esc(task_desc),
            step_count=step_count,
            history_length=min(4, len(history)),
            action_history=_esc(ctx),
            current_step=current_step,
            current_observation=_esc(obs[:2000]),
            previous_belief=_esc(json.dumps(cum_belief, ensure_ascii=False)),
            expert_think=_esc(expert_think),
            expert_action=_esc(action),
        )

        llm_resp = call_llm(client, model, tag_prompt)
        if not llm_resp:
            belief_json_str = json.dumps(cum_belief)
            llm_resp = f"<belief>{belief_json_str}</belief>\n<think>N/A</think>\n<action>{action}</action>"
        else:
            belief_json_str = parse_belief_json(llm_resp) or json.dumps(cum_belief)

        # Post-processing: deterministic prediction + phase transitions
        belief_json_str = _fix_ws_belief_post(belief_json_str, action, next_obs)

        # Fix: confirmed monotonicity — re-inject any keys dropped by LLM
        try:
            b_new = json.loads(belief_json_str)
            cum_confirmed = cum_belief.get('state', {}).get('confirmed', {})
            new_confirmed = b_new.setdefault('state', {}).setdefault('confirmed', {})
            for k, v in cum_confirmed.items():
                if k not in new_confirmed:
                    new_confirmed[k] = v
            belief_json_str = json.dumps(b_new, ensure_ascii=False)
        except Exception:
            pass

        annotated_steps.append({
            'idx': step_idx,
            'obs': obs,
            'admissible': admissible,
            'ground_truth_action': action,
            'next_obs': next_obs,
            'llm_response': llm_resp.strip(),
            'belief_json': belief_json_str,
        })

        try:
            _merge_ws_belief(cum_belief, json.loads(belief_json_str))
        except Exception:
            pass
        history.append(action)

    return {'id': f'ep_{ep_idx}', 'task': task_desc, 'steps': annotated_steps}


def annotate_webshop(input_path: str, annotated_path: str,
                     client: OpenAI, model: str,
                     max_trajectories: int, workers: int):
    _, _, tmpl_tag, _, _ = _ws_templates()

    episodes = []
    if input_path.endswith('.jsonl'):
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
    else:
        with open(input_path) as f:
            episodes = json.load(f)

    if max_trajectories > 0:
        episodes = episodes[:max_trajectories]

    done_ids = _load_done_ep_ids(annotated_path)
    os.makedirs(os.path.dirname(os.path.abspath(annotated_path)), exist_ok=True)
    out_f    = open(annotated_path, 'a')
    out_lock = Lock()

    print(f"Episodes: {len(episodes)}  |  Already done: {len(done_ids)}  |  Workers: {workers}")

    def process_ep(args):
        idx, ep = args
        ep_id = f'ep_{idx}'
        if ep_id in done_ids:
            return None
        result = _annotate_ws_episode(idx, ep, client, model, tmpl_tag)
        if result:
            print(f"  [{idx+1}/{len(episodes)}] task={result['task'][:50]}  steps={len(result['steps'])}")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_ep, (i, ep)): i for i, ep in enumerate(episodes)}
        written = 0
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                with out_lock:
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    out_f.flush()
                    written += 1

    out_f.close()
    print(f"\nStage 1 done. {len(done_ids) + written} trajectories in {annotated_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Convert annotated WebShop → SFT JSONL
# ─────────────────────────────────────────────────────────────────────────────

def convert_webshop(annotated_path: str, output_path: str):
    _, _, _, tmpl_sft_hist, tmpl_sft_nohist = _ws_templates()

    records = []
    with open(annotated_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pairs = []

    for rec in records:
        task_desc = rec['task']
        steps     = rec['steps']

        cum_belief: Dict = {
            'state': {'target': {}, 'confirmed': {}, 'unconfirmed': [], 'product_id': None},
            'task': {'phase': 'searching', 'target': None, 'subgoal': 'Search for target product'},
        }
        history: List[str] = []

        for step in steps:
            step_idx   = step['idx']
            obs        = step['obs']
            admissible = step['admissible']
            gt_action  = step['ground_truth_action']
            llm_resp   = step['llm_response']
            belief_str = step['belief_json']

            action_history = '\n'.join(f"[Step {i+1}: {a}]" for i, a in enumerate(history[-5:]))

            if step_idx == 0:
                sft_prompt = tmpl_sft_nohist.format(
                    task_description=_esc(task_desc),
                    current_observation=_esc(obs[:2000]),
                )
            else:
                sft_prompt = tmpl_sft_hist.format(
                    task_description=_esc(task_desc),
                    step_count=step_idx,
                    history_length=min(5, len(history)),
                    action_history=_esc(action_history),
                    current_step=step_idx + 1,
                    current_observation=_esc(obs[:2000]),
                    previous_belief=_esc(json.dumps(cum_belief, indent=2, ensure_ascii=False)),
                )

            # Deterministically overwrite prediction with ground-truth next_obs
            next_obs = step.get('next_obs', '')
            if next_obs:
                llm_resp = inject_prediction(llm_resp, gt_action, next_obs)

            # Tagging template no longer outputs <action>; append ground-truth action
            if '<action>' not in llm_resp:
                llm_resp = llm_resp.strip() + f'\n\n<action>\n{gt_action}\n</action>'

            pairs.append({'prompt': sft_prompt, 'response': llm_resp})

            try:
                _merge_ws_belief(cum_belief, json.loads(belief_str))
            except Exception:
                pass
            history.append(gt_action)

    with open(output_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"Stage 2 done. {len(pairs)} SFT pairs written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ReBel SFT data generation — two-stage pipeline')
    parser.add_argument('--env',    choices=['alfworld', 'webshop'], required=True)
    parser.add_argument('--stage',  choices=['annotate', 'convert', 'both'], default='both')
    parser.add_argument('--input',  default=None, help='Raw trajectory JSON/JSONL (Stage 1)')
    parser.add_argument('--annotated', required=True,
                        help='Annotated trajectory JSONL (Stage 1 output / Stage 2 input)')
    parser.add_argument('--output', default=None, help='SFT pair JSONL (Stage 2 output)')
    parser.add_argument('--api_base', default='http://localhost:8000/v1')
    parser.add_argument('--api_key',  default='EMPTY')
    parser.add_argument('--model',    default='Qwen/Qwen2.5-72B-Instruct')
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--max_trajectories', type=int, default=0)
    args = parser.parse_args()

    if args.stage in ('annotate', 'both') and not args.input:
        parser.error('--input is required for stage annotate/both')
    if args.stage in ('convert', 'both') and not args.output:
        parser.error('--output is required for stage convert/both')

    api_key = args.api_key
    if api_key == 'EMPTY':
        api_key = (os.environ.get('OPENAI_API_KEY') or
                   os.environ.get('DEEPSEEK_API_KEY') or 'EMPTY')

    client = OpenAI(api_key=api_key, base_url=args.api_base)

    print(f"Env      : {args.env}")
    print(f"Stage    : {args.stage}")
    print(f"API base : {args.api_base}")
    print(f"Model    : {args.model}")
    print(f"Workers  : {args.workers}")
    print(f"Annotated: {args.annotated}")
    if args.output:
        print(f"Output   : {args.output}")

    if args.env == 'alfworld':
        if args.stage in ('annotate', 'both'):
            annotate_alfworld(args.input, args.annotated, client, args.model,
                              args.max_trajectories, args.workers)
        if args.stage in ('convert', 'both'):
            convert_alfworld(args.annotated, args.output)
    else:
        if args.stage in ('annotate', 'both'):
            annotate_webshop(args.input, args.annotated, client, args.model,
                             args.max_trajectories, args.workers)
        if args.stage in ('convert', 'both'):
            convert_webshop(args.annotated, args.output)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
