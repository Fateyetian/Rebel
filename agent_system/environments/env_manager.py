from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import re
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
import copy

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


# ALFWorld task types for V2 task-aware grouping
ALFWORLD_TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def extract_task_type_from_gamefile(gamefile: str) -> str:
    """Extract task type from gamefile path for ReBel V2 task-aware grouping.

    Args:
        gamefile: Path like '/path/to/pick_and_place_simple-Potato-None-...'

    Returns:
        task_type: One of ALFWORLD_TASK_TYPES or 'unknown'
    """
    if not gamefile:
        return "unknown"

    for task in ALFWORLD_TASK_TYPES:
        if task in gamefile:
            return task
    return "unknown"


def extract_phase_gt_alfworld(admissible_commands: list) -> str:
    """Derive current task phase from ALFWorld admissible_commands (game engine output).

    The command set encodes what is currently possible, which is a stable signal
    unlike model-generated belief text. Maps to 4 values: find/pickup/transform/place.
    """
    cmds = ' '.join(admissible_commands).lower() if admissible_commands else ''
    # 'put' only appears when agent is holding an object and near a receptacle
    if 'put' in cmds:
        return 'place'
    # heat/cool/clean indicate the object transformation step
    if any(x in cmds for x in ('heat', 'cool', 'clean', 'slice')):
        return 'transform'
    # 'pick up' available means there is a pickupable object in reach
    if 'pick up' in cmds:
        return 'pickup'
    return 'find'


def extract_phase_gt_webshop(obs: str) -> str:
    """Derive current shopping phase from WebShop observation text markers."""
    if not obs:
        return 'search'
    if '[Buy Now]' in obs or 'buy now' in obs.lower():
        return 'buy'
    if 'Back to Search' in obs or 'Prev Page' in obs or 'Next Page' in obs:
        return 'browse'
    return 'search'


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config

        # Check if ReBel mode is enabled
        self.use_rebel = (
            config is not None and
            hasattr(config, 'algorithm') and
            hasattr(config.algorithm, 'rebel') and
            getattr(config.algorithm.rebel, 'enable', False)
        )

        # V3: Get prompt template type
        self.prompt_template_type = "default"
        if config is not None and hasattr(config, 'env') and hasattr(config.env, 'alfworld'):
            self.prompt_template_type = getattr(config.env.alfworld, 'prompt_template_type', 'default')

        # Initialize ReBel components if enabled
        if self.use_rebel:
            from agent_system.environments.env_package.alfworld import (
                BeliefStateParser, GroundTruthTracker, RebelRewardCalculator
            )
            self.belief_parser = BeliefStateParser()

            # V8 Ablation: Read use_belief_reward and use_result_reward parameters
            use_belief_reward = getattr(config.algorithm.rebel, 'use_belief_reward', True)
            use_result_reward = getattr(config.algorithm.rebel, 'use_result_reward', True)

            # V9: Read belief reward decay parameters
            belief_reward_decay_config = getattr(config.algorithm.rebel, 'belief_reward_decay', None)
            if belief_reward_decay_config is not None:
                belief_reward_decay_enable = getattr(belief_reward_decay_config, 'enable', False)
                belief_reward_decay_method = getattr(belief_reward_decay_config, 'method', 'cosine')
                belief_reward_warmup_epochs = getattr(belief_reward_decay_config, 'warmup_epochs', 5)
                belief_reward_decay_start_epoch = getattr(belief_reward_decay_config, 'decay_start_epoch', 10)
                belief_reward_decay_end_epoch = getattr(belief_reward_decay_config, 'decay_end_epoch', 60)
                belief_reward_min_weight = getattr(belief_reward_decay_config, 'min_weight', 0.1)
            else:
                belief_reward_decay_enable = False
                belief_reward_decay_method = 'cosine'
                belief_reward_warmup_epochs = 5
                belief_reward_decay_start_epoch = 10
                belief_reward_decay_end_epoch = 60
                belief_reward_min_weight = 0.1

            self.reward_calculator = RebelRewardCalculator(
                alpha=getattr(config.algorithm.rebel, 'alpha', 0.40),
                beta=getattr(config.algorithm.rebel, 'beta', 0.30),
                gamma=getattr(config.algorithm.rebel, 'gamma', 0.20),
                delta=getattr(config.algorithm.rebel, 'delta', 0.10),
                use_belief_reward=use_belief_reward,
                belief_reward_decay_enable=belief_reward_decay_enable,
                belief_reward_decay_method=belief_reward_decay_method,
                belief_reward_warmup_epochs=belief_reward_warmup_epochs,
                belief_reward_decay_start_epoch=belief_reward_decay_start_epoch,
                belief_reward_decay_end_epoch=belief_reward_decay_end_epoch,
                belief_reward_min_weight=belief_reward_min_weight,
                # V11: Adaptive decay parameters
                belief_reward_adaptive_decay=getattr(belief_reward_decay_config, 'adaptive', False) if belief_reward_decay_config else False,
                belief_reward_target_sr=getattr(belief_reward_decay_config, 'target_sr', 0.90) if belief_reward_decay_config else 0.90,
                belief_reward_decay_alpha=getattr(belief_reward_decay_config, 'alpha', 2.0) if belief_reward_decay_config else 2.0,
                # V11: Differential component decay rates
                belief_reward_progress_decay_rate=getattr(belief_reward_decay_config, 'progress_decay_rate', 0.7) if belief_reward_decay_config else 0.7,
                belief_reward_consistency_decay_rate=getattr(belief_reward_decay_config, 'consistency_decay_rate', 1.0) if belief_reward_decay_config else 1.0,
                belief_reward_exploration_decay_rate=getattr(belief_reward_decay_config, 'exploration_decay_rate', 2.0) if belief_reward_decay_config else 2.0,
            )

            # Store use_result_reward for later use in reward combination
            self.use_result_reward = use_result_reward

            # Track ground truth for each environment
            self.ground_truth_trackers = {}
            # Track cumulative belief state for each environment
            self.cumulative_beliefs = {}  # env_id -> accumulated belief state
            # Track step count for each environment
            self.step_counts = {}  # env_id -> step count
            # NEW: Track task plans for each environment (filled before first step)
            self.task_plans = {}  # env_id -> parsed planning JSON
        else:
            self.belief_parser = None
            self.reward_calculator = None
            self.ground_truth_trackers = {}
            self.cumulative_beliefs = {}
            self.step_counts = {}
            self.task_plans = {}

        super().__init__(envs, projection_f, env_name)

    def set_current_epoch(self, epoch: int):
        """
        V9: Set current epoch for belief reward decay calculation.
        Should be called by trainer at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        if self.use_rebel and self.reward_calculator is not None:
            self.reward_calculator.set_current_epoch(epoch)
            # Log belief weight for monitoring
            belief_weight = self.reward_calculator.get_belief_reward_weight()
            print(f"[V9] Epoch {epoch}: belief_reward_weight = {belief_weight:.4f}")

    def set_success_rate(self, success_rate: float):
        """
        V11: Set current success rate for adaptive belief reward decay.
        Should be called by trainer after validation.

        Args:
            success_rate: Overall validation success rate (0-1)
        """
        if self.use_rebel and self.reward_calculator is not None:
            self.reward_calculator.set_success_rate(success_rate)
            # Log for monitoring
            belief_weight = self.reward_calculator.get_belief_reward_weight()
            print(f"[V11] Updated success_rate={success_rate:.4f}, belief_reward_weight={belief_weight:.4f}")

    def reset(self, kwargs=None):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)

        # Initialize history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        # Initialize ReBel ground truth trackers
        if self.use_rebel:
            from agent_system.environments.env_package.alfworld import GroundTruthTracker
            self.ground_truth_trackers.clear()
            self.cumulative_beliefs.clear()
            self.step_counts.clear()
            self.task_plans.clear()  # Clear task plans on reset
            # NEW: Initialize history trackers
            self.belief_history = {}  # env_id -> list of predicted belief states
            self.gt_history = {}      # env_id -> list of ground truth states

            for i in range(len(text_obs)):
                tracker = GroundTruthTracker()
                tracker.update_from_observation(text_obs[i])
                self.ground_truth_trackers[i] = tracker

                # Initialize cumulative belief from initial observation (new schema)
                gt_state = tracker.get_ground_truth_state()
                actually_visited = set(gt_state.get('visited', []))

                self.cumulative_beliefs[i] = {
                    'state': {
                        'objects': dict(gt_state.get('object_locations', {})),
                        'states': {},
                        'visited': sorted(actually_visited),
                        'unvisited_candidates': sorted(
                            r for r in gt_state.get('visible_receptacles', [])
                            if r not in actually_visited
                        ),
                    },
                    'task': {
                        'phase': 'find',
                        'target': '',
                        'subgoal': 'Find the target object',
                    },
                }

                # Initialize step count
                self.step_counts[i] = 0

                # NEW: Initialize history lists
                self.belief_history[i] = []
                self.gt_history[i] = []
                # Record initial ground truth state
                self.gt_history[i].append(gt_state.copy())

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)

        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)

        # For ReBel, plannings is actually beliefs
        if self.use_rebel:
            actions, valids, beliefs, action_available = self.projection_f(text_actions, self.envs.get_admissible_commands)
        else:
            actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
            beliefs = [None] * len(text_actions)
            action_available = [True] * len(text_actions)

        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, beliefs)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # Process each environment
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['action_available'] = to_numpy(action_available[i])

            # V2: Extract task_type for task-aware grouping
            gamefile = info.get('extra.gamefile', '')
            info['task_type'] = extract_task_type_from_gamefile(gamefile)

            # GT phase from admissible_commands (stable env signal for hybrid grouping)
            info['gt_phase'] = extract_phase_gt_alfworld(self.envs.get_admissible_commands[i])

            # Calculate ReBel intrinsic rewards if enabled
            if self.use_rebel and i in self.ground_truth_trackers:
                # Update ground truth tracker
                self.ground_truth_trackers[i].update_from_observation(text_obs[i], actions[i])
                ground_truth = self.ground_truth_trackers[i].get_ground_truth_state()

                # NEW: Record ground truth state in history
                if i in self.gt_history:
                    self.gt_history[i].append(ground_truth.copy())

                # Parse belief state from model output
                belief_state = self.belief_parser.parse_belief(full_output[i])

                # NEW: Record predicted belief state in history (even if None)
                if i in self.belief_history:
                    self.belief_history[i].append(belief_state)

                if belief_state:
                    # Calculate intrinsic rewards
                    step = len(self.buffers[i])
                    success = info.get('won', False)

                    # Get format validity flags
                    is_format_valid = bool(valids[i])
                    is_action_available = bool(action_available[i])

                    intrinsic_reward, breakdown = self.reward_calculator.calculate_total_intrinsic_reward(
                        belief_state=belief_state,
                        ground_truth=ground_truth,
                        step=step,
                        done=dones[i],
                        success=success,
                        is_format_valid=is_format_valid,
                        is_action_available=is_action_available,
                        admissible_commands=self.envs.get_admissible_commands[i],
                        next_obs=text_obs[i],
                    )

                    # V8 Ablation: Apply use_result_reward parameter
                    if self.use_result_reward:
                        rewards[i] = float(rewards[i]) + intrinsic_reward
                    else:
                        rewards[i] = intrinsic_reward

                    info['rebel_rewards'] = breakdown
                    info['rebel_intrinsic_reward'] = intrinsic_reward
                    info['belief_state'] = belief_state
                    info['ground_truth_state'] = ground_truth

                    # Update cumulative belief state
                    self._update_cumulative_belief(i, belief_state)
                    # Snapshot cumulative belief for trajectory-aware grouping in advantage computation
                    info['cumulative_belief_state'] = copy.deepcopy(self.cumulative_beliefs.get(i))

                    # Increment step count
                    self.step_counts[i] = self.step_counts.get(i, 0) + 1
                else:
                    # No valid belief state parsed - still give format reward/penalty
                    is_format_valid = bool(valids[i])
                    is_action_available = bool(action_available[i])

                    # Calculate only format reward
                    r_format = self.reward_calculator.calculate_format_reward(
                        is_format_valid=is_format_valid,
                        is_action_available=is_action_available
                    )

                    # Apply format reward
                    intrinsic_reward = self.reward_calculator.delta * r_format

                    # V8 Ablation: Apply use_result_reward parameter
                    if self.use_result_reward:
                        # Normal mode: Add intrinsic reward to environment reward
                        rewards[i] = float(rewards[i]) + intrinsic_reward
                    else:
                        # Ablation mode: Only use intrinsic reward (no result reward)
                        rewards[i] = intrinsic_reward

                    # Store reward breakdown in info
                    info['rebel_rewards'] = {
                        'r_state': 0.0,
                        'r_task': 0.0,
                        'r_pred': 0.0,
                        'r_format': r_format,
                        'r_intrinsic_total': intrinsic_reward
                    }
                    info['rebel_intrinsic_reward'] = intrinsic_reward
                    info['belief_state'] = None
                    info['ground_truth_state'] = ground_truth
                    # Use stale cumulative belief snapshot (current step had invalid format)
                    info['cumulative_belief_state'] = copy.deepcopy(self.cumulative_beliefs.get(i))

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')

            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        # Select template based on ReBel mode
        if self.use_rebel:
            from agent_system.environments.env_package.alfworld.alfworld_rebel_prompt import (
                get_prompt_template
            )
            _ALFWORLD_TEMPLATE_NO_HIS = get_prompt_template(has_history=False)
            _ALFWORLD_TEMPLATE = get_prompt_template(has_history=True)
        else:
            # Default basic template
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            # Get task description
            task_description = self.tasks[i] if i < len(self.tasks) else "Unknown task"

            # V3: Get task_type for explicit_task_type template
            task_type = "unknown"
            if i < len(self.gamefile) and self.gamefile[i]:
                task_type = extract_task_type_from_gamefile(self.gamefile[i])

            # V3: Get current_subgoal for belief_conditioned template
            current_subgoal = "Analyze the task and determine first action"
            if i in self.cumulative_beliefs:
                task_data = self.cumulative_beliefs[i].get('task', {})
                if isinstance(task_data, dict):
                    current_subgoal = task_data.get('subgoal', current_subgoal) or current_subgoal

            if init or history_length <= 0:
                # Initial observation - no history
                if self.use_rebel:
                    obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
                        task_description=task_description,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                    )
                else:
                    obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
            else:
                # With history
                recent_history = self.buffers[i][-history_length:]
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = len(self.buffers[i]) - len(recent_history) + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Step {step_number}: Observation: '{env_obs}', Action: '{action}']"

                if self.use_rebel:
                    # Get cumulative belief state as JSON string for new schema
                    current_belief_json = self._format_belief_state_for_prompt(i)
                    current_step = len(self.buffers[i])
                    step_count = current_step

                    obs = _ALFWORLD_TEMPLATE.format(
                        task_description=task_description,
                        step_count=step_count,
                        history_length=min(history_length, len(self.buffers[i])),
                        action_history=action_history.strip(),
                        current_step=current_step,
                        current_observation=text_obs[i],
                        previous_belief=current_belief_json,
                        admissible_actions=reformatted_admissible_actions,
                    )
                else:
                    obs = _ALFWORLD_TEMPLATE.format(
                        task_description=task_description,
                        step_count=self.step_counts.get(i, 0),
                        history_length=min(history_length, len(self.buffers[i])),
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]),
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions, beliefs=[]):
        for i in range(len(actions)):
            self.buffers[i].append({
                'text_obs': text_obs[i],
                'action': actions[i],
                'full_output': text_actions[i],
                'belief': beliefs[i] if i < len(beliefs) else None
            })

    def _update_cumulative_belief(self, env_id: int, belief_state: Dict[str, Any]):
        """Update cumulative belief state for an environment (new schema)."""
        if env_id not in self.cumulative_beliefs:
            self.cumulative_beliefs[env_id] = {
                'state': {'objects': {}, 'states': {}, 'visited': []},
                'task': {'phase': 'find', 'target': '', 'subgoal': ''},
            }

        cum = self.cumulative_beliefs[env_id]
        state = belief_state.get('state', {}) or {}
        task  = belief_state.get('task',  {}) or {}

        # state.objects: cumulative merge (last-write-wins per object)
        new_objects = state.get('objects', {})
        if isinstance(new_objects, dict) and new_objects:
            cum['state']['objects'].update(new_objects)

        # state.states: cumulative merge
        new_states = state.get('states', {})
        if isinstance(new_states, dict) and new_states:
            cum['state']['states'].update(new_states)

        # state.visited: cumulative deduplicated append
        new_visited = state.get('visited', [])
        if isinstance(new_visited, list):
            existing = set(cum['state']['visited'])
            for v in new_visited:
                if isinstance(v, str) and v and v not in existing:
                    cum['state']['visited'].append(v)
                    existing.add(v)

        # state.unvisited_candidates: add new; remove those now in visited
        new_unvisited = state.get('unvisited_candidates', [])
        if isinstance(new_unvisited, list):
            visited_set = set(cum['state']['visited'])
            existing_uv = set(cum['state'].get('unvisited_candidates', []))
            for loc in new_unvisited:
                if isinstance(loc, str) and loc and loc not in visited_set and loc not in existing_uv:
                    cum['state'].setdefault('unvisited_candidates', []).append(loc)
                    existing_uv.add(loc)
            cum['state']['unvisited_candidates'] = [
                u for u in cum['state'].get('unvisited_candidates', [])
                if u not in visited_set
            ]

        # task: carry latest non-empty values
        for key in ('phase', 'target', 'subgoal'):
            val = task.get(key)
            if val and isinstance(val, str):
                cum['task'][key] = val

    def _format_belief_state_for_prompt(self, env_id: int) -> str:
        """Return cumulative belief state as JSON string for prompt injection."""
        if env_id not in self.cumulative_beliefs:
            return '{}'
        import json
        return json.dumps(self.cumulative_beliefs[env_id], indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW: Planning Support Methods - Run BEFORE main interaction loop
    # ═══════════════════════════════════════════════════════════════════════════════

    def get_planning_prompts(self) -> List[str]:
        """
        Get planning prompts for all environments.
        Should be called BEFORE the main interaction loop starts.
        Returns a list of planning prompts, one for each environment.
        """
        if not self.use_rebel:
            return []

        from agent_system.environments.env_package.alfworld import ALFWORLD_PLANNING_PROMPT_REBEL

        planning_prompts = []
        for i in range(len(self.tasks)):
            # Format the planning prompt with task description and initial observation
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in self.envs.get_admissible_commands[i] if s != 'help'
            )

            prompt = ALFWORLD_PLANNING_PROMPT_REBEL.format(
                task_description=self.tasks[i],
                observation=self.pre_text_obs[i],
                admissible_actions=reformatted_admissible_actions
            )
            planning_prompts.append(prompt)

        return planning_prompts

    def set_task_plans(self, plans: List[dict]):
        """
        Set task plans for all environments.
        Should be called after running planning prompts through the model.

        Args:
            plans: List of parsed planning JSON dicts, one per environment.
                   Each dict should have: main_goal, target_objects, target_receptacle,
                   likely_locations, plan_steps, success_criteria
        """
        if not self.use_rebel:
            return

        for i, plan in enumerate(plans):
            if plan is not None and isinstance(plan, dict):
                self.task_plans[i] = plan

                # Also update cumulative beliefs with planning information
                if i in self.cumulative_beliefs:
                    task_progress = self.cumulative_beliefs[i].get('task_progress', {})
                    task_progress['main_goal'] = plan.get('main_goal', '')
                    task_progress['plan'] = plan.get('plan_steps', [])
                    task_progress['target_objects'] = plan.get('target_objects', [])
                    task_progress['target_receptacle'] = plan.get('target_receptacle', '')
                    task_progress['success_criteria'] = plan.get('success_criteria', '')

                    # Set first subgoal from plan
                    plan_steps = plan.get('plan_steps', [])
                    if plan_steps and len(plan_steps) > 0:
                        first_step = plan_steps[0]
                        if isinstance(first_step, dict):
                            task_progress['subgoal'] = first_step.get('subgoal', '')
                        elif isinstance(first_step, str):
                            task_progress['subgoal'] = first_step

                    self.cumulative_beliefs[i]['task_progress'] = task_progress

                    # Also set exploration priorities from likely_locations
                    likely_locs = plan.get('likely_locations', [])
                    if likely_locs:
                        self.cumulative_beliefs[i]['exploration_map']['unexplored'] = set(likely_locs)

    def has_task_plans(self) -> bool:
        """Check if task plans have been set."""
        return len(self.task_plans) > 0

    def parse_planning_output(self, outputs: List[str]) -> List[dict]:
        """
        Parse model outputs from planning prompts into structured plan dicts.

        Args:
            outputs: List of model output strings (should be JSON)

        Returns:
            List of parsed plan dicts (or None for failed parses)
        """
        import json
        import re

        plans = []
        for output in outputs:
            try:
                # Try to extract JSON from the output
                # First, try direct JSON parse
                plan = json.loads(output.strip())
                plans.append(plan)
            except json.JSONDecodeError:
                # Try to find JSON block in the output
                json_match = re.search(r'\{[\s\S]*\}', output)
                if json_match:
                    try:
                        plan = json.loads(json_match.group())
                        plans.append(plan)
                    except json.JSONDecodeError:
                        plans.append(None)
                else:
                    plans.append(None)

        return plans


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SciWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        self.plannings = []
        super().__init__(envs, projection_f, env_name)

    def reset(self, kwargs=None):
        text_obs, infos = self.envs.reset()

        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos], init=True)
        return {'text': full_text_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        actions, valids, action_available = self.projection_f(text_actions, meta_think=meta_think, available_actions=self.envs.get_possible_actions)

        plannings = []
        if meta_think:
            for action in text_actions:
                planning = None
                if "<planning>" in action and "</planning>" in action:
                    start_tag = "<planning>"
                    end_tag = "</planning>"
                    start_idx = action.find(start_tag)
                    end_idx = action.find(end_tag)
                    if start_idx != -1 and end_idx != -1:
                        planning = action[start_idx + len(start_tag):end_idx].strip()
                plannings.append(planning)
        else:
            plannings = [None] * len(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos])

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['full_output'] = full_output[i]
            info['action_available'] = to_numpy(action_available[i])
            info['score'] = info.get('score', -1)

        next_observations = {'text': full_text_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")

    def build_text_obs(self, text_obs: List[str], available_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if self.meta_think:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_MC
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_MC
        else:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE

        for i in range(len(text_obs)):
            if init or history_length <= 0:
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )
            else:
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""

                for j in range(recent_start_index):
                    action = all_actions[j]
                    step_number = j + 1
                    action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"

                for j, record in enumerate(recent_history):
                    step_number = recent_start_index + j + 1
                    env_obs = record["text_obs"]
                    action = record["action"]
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                if self.config is not None and hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') and self.config.env.sciworld.meta_think:
                    history_think_length = min(3, len(self.buffers[i]))
                    start_index = len(self.buffers[i]) - history_think_length
                    action_history += "\n- recent reasoning process: \n" 
                    for j, record in enumerate(self.buffers[i][-history_think_length:]):
                        step_number = start_index + j + 1
                        action_history += f"[Step {step_number}, output {step_number}: '{record['full_output']}']\n"

                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        planning=self.plannings[i],
                        available_actions=available_actions[i]
                    )
                else:
                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        available_actions=available_actions[i]
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions=None, plannings=None):
        for i in range(len(actions)):
            if text_actions:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
            else:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

        if plannings:
            for i in range(len(plannings)):
                if plannings[i] is not None:
                    self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def _set_meta_think(self, type: bool):
        self.meta_think = type

class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config

        # Check if ReBel mode is enabled
        self.use_rebel = (
            config is not None and
            hasattr(config, 'algorithm') and
            hasattr(config.algorithm, 'rebel') and
            getattr(config.algorithm.rebel, 'enable', False)
        )

        # Initialize ReBel components if enabled
        if self.use_rebel:
            from agent_system.environments.env_package.webshop import (
                WebShopBeliefStateParser, WebShopGroundTruthTracker, WebShopRebelRewardCalculator
            )
            self.belief_parser = WebShopBeliefStateParser()

            # V8 Ablation: Read use_belief_reward and use_result_reward parameters
            use_belief_reward = getattr(config.algorithm.rebel, 'use_belief_reward', True)
            use_result_reward = getattr(config.algorithm.rebel, 'use_result_reward', True)

            # V9: Read belief reward decay parameters
            belief_reward_decay_config = getattr(config.algorithm.rebel, 'belief_reward_decay', None)
            if belief_reward_decay_config is not None:
                belief_reward_decay_enable = getattr(belief_reward_decay_config, 'enable', False)
                belief_reward_decay_method = getattr(belief_reward_decay_config, 'method', 'cosine')
                belief_reward_warmup_epochs = getattr(belief_reward_decay_config, 'warmup_epochs', 5)
                belief_reward_decay_start_epoch = getattr(belief_reward_decay_config, 'decay_start_epoch', 10)
                belief_reward_decay_end_epoch = getattr(belief_reward_decay_config, 'decay_end_epoch', 60)
                belief_reward_min_weight = getattr(belief_reward_decay_config, 'min_weight', 0.1)
            else:
                belief_reward_decay_enable = False
                belief_reward_decay_method = 'cosine'
                belief_reward_warmup_epochs = 5
                belief_reward_decay_start_epoch = 10
                belief_reward_decay_end_epoch = 60
                belief_reward_min_weight = 0.1

            self.reward_calculator = WebShopRebelRewardCalculator(
                alpha=getattr(config.algorithm.rebel, 'alpha', 0.40),
                beta=getattr(config.algorithm.rebel, 'beta', 0.30),
                gamma=getattr(config.algorithm.rebel, 'gamma', 0.20),
                delta=getattr(config.algorithm.rebel, 'delta', 0.10),
                use_belief_reward=use_belief_reward,
                belief_reward_decay_enable=belief_reward_decay_enable,
                belief_reward_decay_method=belief_reward_decay_method,
                belief_reward_warmup_epochs=belief_reward_warmup_epochs,
                belief_reward_decay_start_epoch=belief_reward_decay_start_epoch,
                belief_reward_decay_end_epoch=belief_reward_decay_end_epoch,
                belief_reward_min_weight=belief_reward_min_weight,
                # V11: Adaptive decay parameters
                belief_reward_adaptive_decay=getattr(belief_reward_decay_config, 'adaptive', False) if belief_reward_decay_config else False,
                belief_reward_target_sr=getattr(belief_reward_decay_config, 'target_sr', 0.90) if belief_reward_decay_config else 0.90,
                belief_reward_decay_alpha=getattr(belief_reward_decay_config, 'alpha', 2.0) if belief_reward_decay_config else 2.0,
                # V11: Differential component decay rates
                belief_reward_progress_decay_rate=getattr(belief_reward_decay_config, 'progress_decay_rate', 0.7) if belief_reward_decay_config else 0.7,
                belief_reward_consistency_decay_rate=getattr(belief_reward_decay_config, 'consistency_decay_rate', 1.0) if belief_reward_decay_config else 1.0,
                belief_reward_exploration_decay_rate=getattr(belief_reward_decay_config, 'exploration_decay_rate', 2.0) if belief_reward_decay_config else 2.0,
            )

            self.use_result_reward = use_result_reward
            self.ground_truth_trackers = {}
            self.cumulative_beliefs = {}
            self.step_counts = {}
            self.task_plans = {}
        else:
            self.belief_parser = None
            self.reward_calculator = None
            self.ground_truth_trackers = {}
            self.cumulative_beliefs = {}
            self.step_counts = {}
            self.task_plans = {}

        super().__init__(envs, projection_f, env_name)

    def set_current_epoch(self, epoch: int):
        """V9: Set current epoch for belief reward decay calculation."""
        if self.use_rebel and self.reward_calculator is not None:
            self.reward_calculator.set_current_epoch(epoch)
            belief_weight = self.reward_calculator.get_belief_reward_weight()
            print(f"[V9] Epoch {epoch}: belief_reward_weight = {belief_weight:.4f}")

    def set_success_rate(self, success_rate: float):
        """V11: Set current success rate for adaptive belief reward decay."""
        if self.use_rebel and self.reward_calculator is not None:
            self.reward_calculator.set_success_rate(success_rate)
            belief_weight = self.reward_calculator.get_belief_reward_weight()
            print(f"[V11] Updated success_rate={success_rate:.4f}, belief_reward_weight={belief_weight:.4f}")

    # ------------------------------------------------------------------
    # ReBel Planning interface (mirrors AlfWorldEnvironmentManager)
    # ------------------------------------------------------------------

    def get_planning_prompts(self) -> List[str]:
        """
        Get planning prompts for all WebShop environments.
        Called BEFORE the main interaction loop starts.
        """
        if not self.use_rebel:
            return []

        planning_prompts = []
        for i in range(len(self.tasks)):
            # NOTE: 【Task Description】 marker is required by TeacherPlanner's cache
            # key extraction regex. Do NOT remove or rename this marker.
            prompt = (
                f"You are an expert online shopping agent.\n\n"
                f"【Task Description】\n{self.tasks[i]}\n\n"
                f"Current observation:\n{self.pre_text_obs[i]}\n\n"
                f"Create a step-by-step plan as JSON with keys: "
                f"main_goal, target_attributes, search_strategy, plan_steps, success_criteria."
            )
            planning_prompts.append(prompt)

        return planning_prompts

    def set_task_plans(self, plans: List[dict]):
        """
        Store task plans produced by the teacher / training model.
        """
        if not self.use_rebel:
            return

        for i, plan in enumerate(plans):
            if plan is not None and isinstance(plan, dict):
                self.task_plans[i] = plan

                # Propagate planning info into cumulative_beliefs
                if i in self.cumulative_beliefs:
                    sp = self.cumulative_beliefs[i].get('search_progress', {})
                    sp['main_goal'] = plan.get('main_goal', '')
                    sp['plan'] = plan.get('plan_steps', [])

                    plan_steps = plan.get('plan_steps', [])
                    if plan_steps:
                        first = plan_steps[0]
                        if isinstance(first, dict):
                            sp['updated_subgoal'] = first.get('subgoal', sp.get('updated_subgoal', ''))
                        elif isinstance(first, str):
                            sp['updated_subgoal'] = first

                    self.cumulative_beliefs[i]['search_progress'] = sp

    def has_task_plans(self) -> bool:
        return len(self.task_plans) > 0

    def parse_planning_output(self, outputs: List[str]) -> List[dict]:
        """Parse model outputs from planning prompts into structured plan dicts."""
        import json as _json
        plans = []
        for output in outputs:
            try:
                plan = _json.loads(output.strip())
                plans.append(plan)
            except _json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*\}', output)
                if json_match:
                    try:
                        plan = _json.loads(json_match.group())
                        plans.append(plan)
                    except _json.JSONDecodeError:
                        plans.append(None)
                else:
                    plans.append(None)
        return plans

    def reset(self, kwargs=None) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        observations = {'text': self.build_text_obs(obs, infos, init=True),
                        'image': None,
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]

        # Initialize ReBel ground truth trackers
        if self.use_rebel:
            from agent_system.environments.env_package.webshop import WebShopGroundTruthTracker
            self.ground_truth_trackers.clear()
            self.cumulative_beliefs.clear()
            self.step_counts.clear()

            for i in range(len(infos)):
                tracker = WebShopGroundTruthTracker()
                tracker.task_goal = self.tasks[i]
                tracker.update_from_observation(obs[i], info=infos[i])
                self.ground_truth_trackers[i] = tracker

                self.cumulative_beliefs[i] = {
                    'state': {
                        'target': {},
                        'confirmed': {},
                        'unconfirmed': [],
                        'product_id': None,
                    },
                    'task': {
                        'phase': 'searching',
                        'target': None,
                        'subgoal': 'Search for the target product',
                    },
                }
                self.step_counts[i] = 0
        elif hasattr(self, 'belief_mgr'):
            # initialize belief states with tasks (original logic)
            env_ids = list(range(len(infos)))
            try:
                self.belief_mgr.reset(env_ids=env_ids, task_desc=self.tasks)
            except Exception:
                self.belief_mgr.reset(env_ids=env_ids, task_desc=None)

        # Store infos for first step's projection (needed for available actions in ReBel)
        self._last_infos = infos

        return observations, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)

        if self.use_rebel:
            # ReBel mode: 4-value projection
            avail_actions = [info.get('available_actions', {}) if isinstance(info, dict) else {}
                           for info in self._last_infos] if hasattr(self, '_last_infos') else [{}] * len(text_actions)
            actions, valids, beliefs, action_available = self.projection_f(text_actions, avail_actions)
        else:
            actions, valids = self.projection_f(text_actions)
            beliefs = [''] * len(actions)
            action_available = [False] * len(actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_obs = self.format_obs(next_obs)

        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, beliefs)
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }

        # Process each environment
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['action_available'] = to_numpy(action_available[i])

            # GT phase from observation markers (stable env signal for hybrid grouping)
            info['gt_phase'] = extract_phase_gt_webshop(next_obs[i] if i < len(next_obs) else '')

            if self.use_rebel and i in self.ground_truth_trackers:
                # Update ground truth tracker
                self.ground_truth_trackers[i].update_from_observation(next_obs[i], actions[i], info)
                ground_truth = self.ground_truth_trackers[i].get_ground_truth_state()

                # Parse belief state from model output
                belief_state = self.belief_parser.parse_belief(full_output[i])

                if belief_state:
                    step = len(self.buffers[i])
                    success = info.get('won', False)
                    is_format_valid = bool(valids[i])
                    is_action_available = bool(action_available[i])

                    intrinsic_reward, breakdown = self.reward_calculator.calculate_total_intrinsic_reward(
                        belief_state=belief_state,
                        ground_truth=ground_truth,
                        step=step,
                        done=dones[i],
                        success=success,
                        is_format_valid=is_format_valid,
                        is_action_available=is_action_available,
                        current_obs=next_obs[i],
                        next_obs=next_obs[i],
                    )

                    if self.use_result_reward:
                        rewards[i] = float(rewards[i]) + intrinsic_reward
                    else:
                        rewards[i] = intrinsic_reward

                    info['rebel_rewards'] = breakdown
                    info['rebel_intrinsic_reward'] = intrinsic_reward
                    info['belief_state'] = belief_state
                    info['ground_truth_state'] = ground_truth

                    self._update_cumulative_belief(i, belief_state)
                    info['cumulative_belief_state'] = copy.deepcopy(self.cumulative_beliefs.get(i))
                    self.step_counts[i] = self.step_counts.get(i, 0) + 1
                else:
                    is_format_valid = bool(valids[i])
                    is_action_available = bool(action_available[i])
                    r_format = self.reward_calculator.calculate_format_reward(
                        is_format_valid=is_format_valid,
                        is_action_available=is_action_available
                    )
                    intrinsic_reward = self.reward_calculator.delta * r_format

                    if self.use_result_reward:
                        rewards[i] = float(rewards[i]) + intrinsic_reward
                    else:
                        rewards[i] = intrinsic_reward

                    info['rebel_rewards'] = {
                        'r_state': 0.0,
                        'r_task': 0.0,
                        'r_pred': 0.0,
                        'r_format': r_format,
                        'r_intrinsic_total': intrinsic_reward
                    }
                    info['rebel_intrinsic_reward'] = intrinsic_reward
                    info['belief_state'] = None
                    info['ground_truth_state'] = ground_truth
                    info['cumulative_belief_state'] = copy.deepcopy(self.cumulative_beliefs.get(i))
            elif not self.use_rebel:
                # Original belief manager logic
                try:
                    anchor_obs = self.pre_text_obs[i]
                    self.belief_mgr.step_update(
                        env_id=i, observation=anchor_obs,
                        action=actions[i], info=info, reward=float(rewards[i]),
                    )
                    snap = self.belief_mgr.snapshot(i)
                    info['belief'] = {
                        'step_idx': snap.step_idx,
                        'world_model': snap.world_model,
                        'task_progress': snap.task_progress,
                        'exploration_map': {
                            'visited_rooms': list(snap.exploration_map.get('visited_rooms', [])),
                            'visited_objects': list(snap.exploration_map.get('visited_objects', [])),
                        },
                        'notes': snap.notes,
                    }
                except Exception:
                    pass

        # Store infos for next step's projection (needed for available actions)
        self._last_infos = infos

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def _update_cumulative_belief(self, env_id: int, belief_state: Dict[str, Any]):
        """Update cumulative belief state from parsed belief (new schema)."""
        if env_id not in self.cumulative_beliefs:
            return

        cum = self.cumulative_beliefs[env_id]
        state = belief_state.get('state', {}) or {}
        task  = belief_state.get('task',  {}) or {}

        # state.target: merge (task requirements, rarely changes)
        new_target = state.get('target', {})
        if isinstance(new_target, dict) and new_target:
            cum['state']['target'].update(new_target)

        # state.confirmed: cumulative merge (verified attributes)
        new_confirmed = state.get('confirmed', {})
        if isinstance(new_confirmed, dict) and new_confirmed:
            cum['state']['confirmed'].update(new_confirmed)
            # Remove from unconfirmed once confirmed
            confirmed_keys = set(cum['state']['confirmed'].keys())
            cum['state']['unconfirmed'] = [
                u for u in cum['state']['unconfirmed'] if u not in confirmed_keys
            ]

        # state.unconfirmed: append new attrs not yet confirmed
        new_unconfirmed = state.get('unconfirmed', [])
        if isinstance(new_unconfirmed, list):
            confirmed_keys = set(cum['state']['confirmed'].keys())
            existing = set(cum['state']['unconfirmed'])
            for attr in new_unconfirmed:
                if isinstance(attr, str) and attr and attr not in existing and attr not in confirmed_keys:
                    cum['state']['unconfirmed'].append(attr)
                    existing.add(attr)

        # state.product_id: latest non-null value
        new_pid = state.get('product_id')
        if new_pid and new_pid != 'null':
            cum['state']['product_id'] = str(new_pid)

        # task: carry latest non-empty values
        for key in ('phase', 'target', 'subgoal'):
            val = task.get(key)
            if val is not None and str(val).strip():
                cum['task'][key] = val

    def _format_belief_state_for_prompt(self, env_id: int) -> str:
        """Format cumulative belief state for inclusion in prompts"""
        if env_id not in self.cumulative_beliefs:
            return "No belief state available."

        cum = self.cumulative_beliefs[env_id]
        import json
        return json.dumps(cum, indent=2, ensure_ascii=False)

    @staticmethod
    def _compress_webshop_obs(obs: str, desc_limit: int = 300) -> str:
        """
        Segment-aware compression of a WebShop observation.

        WebShop obs is `[SEP]`-delimited visible HTML text.  The structure is:
          nav | Instruction: | task | title | price | option-buttons | Description |
          <long marketing copy> | Features | <feature list> | Reviews | <reviews>

        Only the long unstructured segments (description body, feature lists,
        reviews) need compression — everything else is already short and is
        preserved verbatim.  This keeps the model's full view of product title,
        price, and available clickable options while eliminating the token
        waste from marketing copy.

        Args:
            obs:        Raw [SEP]-delimited WebShop observation string.
            desc_limit: Max chars per long segment before truncating.

        Returns:
            Compressed observation string (same [SEP] structure).
        """
        SEP = ' [SEP] '
        parts = obs.split(SEP)

        # Section headers that signal the START of a long-form content block.
        # Segments immediately after these are the verbose ones.
        _VERBOSE_HEADERS = {'description', 'features', 'reviews', 'product description'}

        compressed = []
        next_is_verbose = False
        for part in parts:
            stripped = part.strip()
            if stripped.lower() in _VERBOSE_HEADERS:
                compressed.append(part)
                next_is_verbose = True
            elif next_is_verbose and len(stripped) > desc_limit:
                # Truncate the verbose block; mark so the model knows content continues
                compressed.append(stripped[:desc_limit] + ' ...')
                next_is_verbose = False
            else:
                compressed.append(part)
                # A short segment resets the verbose flag (we've moved to the next section)
                if len(stripped) < 80:
                    next_is_verbose = False

        return SEP.join(compressed)

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks

    def format_obs(self, text_obs):
        # Return raw observations as-is to match SFT training data format.
        # SFT data uses the raw WebShop observation strings without reformatting.
        return list(text_obs)

    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def save_to_history_buffer(self, text_obs, actions, full_output=None, beliefs=None):
        for i in range(len(actions)):
            self.buffers[i].append({
                'text_obs': text_obs[i],
                'action': actions[i],
                'full_output': full_output[i] if full_output else "",
                'belief': beliefs[i] if beliefs and i < len(beliefs) else None
            })

    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False, history_length: int = 5) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        # Select template based on ReBel mode
        if self.use_rebel:
            from agent_system.environments.prompts.webshop_rebel_prompts import (
                WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL,
                WEBSHOP_REBEL_TEMPLATE_RL
            )
            _TEMPLATE_NO_HIS = WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL
            _TEMPLATE = WEBSHOP_REBEL_TEMPLATE_RL
        else:
            _TEMPLATE_NO_HIS = WEBSHOP_TEMPLATE_NO_HIS
            _TEMPLATE = WEBSHOP_TEMPLATE

        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or history_length <= 0:
                if self.use_rebel:
                    obs = _TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_available_actions
                    )
                else:
                    obs = _TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    if self.use_rebel:
                        # ReBel mode: actions-only history. The cumulative belief state already
                        # encodes what was observed (exploration_state tracks queries_tried,
                        # products_viewed, options_selected, tabs_clicked). Including full
                        # WebShop page observations (500-1500 tokens each) is redundant and
                        # causes prompt length to exceed token limits.
                        action_history += f"\n[Step {step_number}: {action}]"
                    else:
                        action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                if self.use_rebel:
                    # Get cumulative belief state for ReBel
                    # Compress obs before building prompt: only verbose description/review
                    # segments are trimmed; title, price, options stay intact.
                    prompt_obs = self._compress_webshop_obs(text_obs[i])
                    belief_state_str = self._format_belief_state_for_prompt(i)

                    obs = _TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=prompt_obs,
                        admissible_actions=reformatted_available_actions,
                        previous_belief=belief_state_str,
                    )
                    # Fallback only if compressed obs is still too large (rare edge case)
                    if len(obs) > 13000:
                        import json as _json
                        compact_belief = _json.dumps(
                            {'task': self.cumulative_beliefs.get(i, {}).get('task', {})},
                            ensure_ascii=False
                        )
                        obs = _TEMPLATE.format(
                            task_description=self.tasks[i],
                            step_count=len(self.buffers[i]),
                            history_length=valid_history_length,
                            action_history=action_history.strip(),
                            current_step=len(self.buffers[i]) + 1,
                            current_observation=prompt_obs,
                            admissible_actions=reformatted_available_actions,
                            previous_belief=compact_belief,
                        )
                else:
                    obs = _TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
                    if len(obs) > 13000:
                        print(f"Warning len(obs)={len(obs)} is too long")
                        obs = _TEMPLATE_NO_HIS.format(
                            task_description=self.tasks[i],
                            current_observation=text_obs[i],
                            available_actions=reformatted_available_actions
                        )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info.get('task_score', 0.0))
                # success_rate = score == 1.0 (GiGPO standard, fair comparison)
                success['success_rate'].append(float(score_value >= 1.0 - 1e-6))
                success['webshop_task_score (not success_rate)'].append(score_value)
                # Finer-grained score buckets for diagnosing reward distribution
                success['webshop_score_ge_0.3'].append(float(score_value >= 0.3))
                success['webshop_score_ge_0.5'].append(float(score_value >= 0.5))
                success['webshop_score_ge_0.8'].append(float(score_value >= 0.8))
                success['webshop_score_eq_1.0'].append(float(score_value >= 1.0 - 1e-6))
                # Track whether the episode actually terminated (agent clicked buy now)
                last_info = total_infos[batch_idx][i]
                success['webshop_bought'].append(float(last_info.get('task_score', 0.0) > 0))
                return


def make_envs(config):
    """
    Create enviroments
    """
    print("[DEBUG make_envs] Starting make_envs()")
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    print(f"[DEBUG make_envs] group_n={group_n}, env_name={config.env.env_name}")

    if "alfworld" in config.env.env_name.lower():
        print("[DEBUG make_envs] Creating ALFWorld environments")
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection, alfworld_projection_rebel

        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        print(f"[DEBUG make_envs] alf_config_path={alf_config_path}")
        print(f"[DEBUG make_envs] generalization_level={config.env.alfworld.generalization_level}")

        if config.env.alfworld.generalization_level == 2:
            alf_train_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_train_ood.yaml')
            alf_test_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_test_ood.yaml')
            print(f"[DEBUG make_envs] Building training envs (level 2)...")
            train_env_num = config.data.train_batch_size if hasattr(config, 'trainer') and config.trainer.total_epochs > 0 else group_n
            # Fix: Use val_batch_size for validation environments, not group_n
            val_env_num = config.data.val_batch_size if hasattr(config.data, 'val_batch_size') else group_n
            _envs = build_alfworld_envs(alf_train_config_path, config.env.seed, train_env_num, group_n, is_train=True)
            print(f"[DEBUG make_envs] Building validation envs (level 2)...")
            _val_envs = build_alfworld_envs(alf_test_config_path, config.env.seed + 1000, val_env_num, 1, is_train=False, unseen=True)
        elif config.env.alfworld.generalization_level == 1:
            print(f"[DEBUG make_envs] Building training envs (level 1)...")
            train_env_num = config.data.train_batch_size if hasattr(config, 'trainer') and config.trainer.total_epochs > 0 else group_n
            # Fix: Use val_batch_size for validation environments, not group_n
            val_env_num = config.data.val_batch_size if hasattr(config.data, 'val_batch_size') else group_n
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, train_env_num, group_n, is_train=True)
            print(f"[DEBUG make_envs] Building validation envs (level 1)...")
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, val_env_num, 1, is_train=False, unseen=True)
        elif config.env.alfworld.generalization_level == 0:
            print(f"[DEBUG make_envs] Building training envs (level 0)...")
            # Use env.rollout.n for validation-only mode when total_epochs=0
            train_env_num = config.data.train_batch_size if hasattr(config, 'trainer') and config.trainer.total_epochs > 0 else group_n
            # Fix: Use val_batch_size for validation environments, not group_n
            val_env_num = config.data.val_batch_size if hasattr(config.data, 'val_batch_size') else group_n
            print(f"[DEBUG make_envs] train_env_num={train_env_num}, val_env_num={val_env_num}")
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, train_env_num, group_n, is_train=True)
            print(f"[DEBUG make_envs] Training envs built successfully")
            print(f"[DEBUG make_envs] Building validation envs (level 0)...")
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, val_env_num, 1, is_train=False)
            print(f"[DEBUG make_envs] Validation envs built successfully")

        print(f"[DEBUG make_envs] Setting up projection function...")
        # Check if ReBel is enabled
        use_rebel = (hasattr(config, 'algorithm') and
                    hasattr(config.algorithm, 'rebel') and
                    getattr(config.algorithm.rebel, 'enable', False))

        if use_rebel:
            print("[DEBUG make_envs] Using ReBel projection (<belief>/<action>)")
            projection_f = partial(alfworld_projection_rebel)
        else:
            print("[DEBUG make_envs] Using default projection (<think>/<action>)")
            projection_f = partial(alfworld_projection)

        print(f"[DEBUG make_envs] Creating AlfWorldEnvironmentManager for training...")
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        print(f"[DEBUG make_envs] Creating AlfWorldEnvironmentManager for validation...")
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)
        print(f"[DEBUG make_envs] Environment managers created successfully")
        return envs, val_envs
    elif "sciworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
        import json
        generalization_level = config.env.sciworld['generalization_level']

        if generalization_level == 2:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L2_idx.json'
        elif generalization_level == 1:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L1_idx.json'
        elif generalization_level == 0:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'

        with open(variation_path, 'r') as f:
            variations_idx = json.load(f)

        simplifications_preset = config.env.sciworld.get('simplifications_preset', "easy")
        env_step_limit = config.env.sciworld.get('env_step_limit', 100)
        jar_path = config.env.sciworld.get('jar_path', None)

        _envs = build_sciworld_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['train']
        )

        _val_envs = build_sciworld_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['test']
        )

        # Create projection function
        projection_f = partial(sciworld_projection)

        # Create environment managers
        envs = SciWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        val_envs = SciWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)

        # Give some time for environments to initialize
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)

        return envs, val_envs

    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection, webshop_projection_rebel
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)

        # Check if ReBel is enabled
        use_rebel = (hasattr(config, 'algorithm') and
                    hasattr(config.algorithm, 'rebel') and
                    getattr(config.algorithm.rebel, 'enable', False))

        if use_rebel:
            print("[DEBUG make_envs] WebShop: Using ReBel projection (<belief>/<action>)")
            projection_f = partial(webshop_projection_rebel)
        else:
            print("[DEBUG make_envs] WebShop: Using default projection (<think>/<action>)")
            projection_f = partial(webshop_projection)

        envs = WebshopEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)