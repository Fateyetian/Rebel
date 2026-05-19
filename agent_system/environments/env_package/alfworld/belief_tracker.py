"""
ReBel (Reward Belief) Framework - ALFWorld Belief Tracking and Intrinsic Reward

Three reward signals (paper §4.2):
  r_state : belief.state predicates vs GroundTruthTracker (accumulated observations)
  r_task  : belief.task.phase vs env GT phase from admissible_commands
  r_pred  : belief.prediction keyword match vs o_{t+1}
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# GT phase extraction from admissible_commands
# ─────────────────────────────────────────────────────────────────────────────

def extract_phase_gt_alfworld(admissible_commands: List[str], holding: bool = False) -> str:
    """
    Derive ground-truth task phase from ALFWorld admissible_commands.

    Priority order matches ALFWorld task structure:
    transform > place > pickup > find/navigate
    """
    if not admissible_commands:
        return 'find'
    cmds = ' '.join(admissible_commands).lower()

    if any(k in cmds for k in ['heat ', 'cool ', 'clean ']):
        return 'transform'
    if 'put ' in cmds and holding:
        return 'place'
    if 'pick up' in cmds:
        return 'pickup'
    if 'go to' in cmds:
        return 'find'
    return 'find'


# ─────────────────────────────────────────────────────────────────────────────
# r_pred: prediction keyword match vs next observation
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    'that', 'will', 'the', 'and', 'or', 'to', 'a', 'an', 'in', 'on', 'at',
    'be', 'see', 'find', 'expect', 'next', 'observation', 'if', 'i', 'you',
    'then', 'with', 'from', 'this', 'it', 'is', 'are', 'was', 'not', 'no',
}

# High-frequency action words that appear in many obs regardless of prediction.
# Matching them as singletons produces false positives (e.g. "closed" in any
# close-action obs).  They are still useful inside bigrams.
_HIGH_NOISE_UNIGRAMS = {
    'closed', 'open', 'opened', 'pick', 'picked', 'found', 'clean', 'cleaned',
    'heat', 'heated', 'cool', 'cooled', 'put', 'place', 'placed', 'hold',
    'holding', 'examine', 'look', 'move', 'moved', 'take', 'taken',
}


def compute_r_pred(prediction: str, next_obs: str) -> float:
    """
    Bigram-anchored keyword match between belief.prediction and o_{t+1}.

    Extracts the clause after "I expect", builds (unigram + bigram) feature
    sets, then scores against next_obs.  High-noise singleton words (e.g.
    "closed") are only counted when they appear as part of a matched bigram,
    preventing false positives from generic action descriptions.
    Returns [0, 1].
    """
    if not prediction or not next_obs:
        return 0.0

    pred_lower = prediction.lower()
    obs_lower  = next_obs.lower()

    # Pull out the expected-content clause
    m = re.search(r'i expect\s+(.*)', pred_lower)
    expected = m.group(1) if m else pred_lower

    # All content tokens (length > 3, not stopwords)
    tokens = [
        w for w in re.findall(r'\b[a-z]\w+\b', expected)
        if len(w) > 3 and w not in _STOPWORDS
    ]
    if not tokens:
        return 0.0

    # Build unigrams (excluding high-noise singletons) + bigrams
    features: list = []
    for i, tok in enumerate(tokens):
        if tok not in _HIGH_NOISE_UNIGRAMS:
            features.append(tok)          # safe unigram
        if i < len(tokens) - 1:
            features.append(f'{tok} {tokens[i + 1]}')  # bigram (always added)

    if not features:
        # Fallback: prediction consists entirely of high-noise words — use them
        features = tokens

    matches = sum(1 for feat in features if feat in obs_lower)
    return matches / len(features)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class BeliefStateParser:
    """Parse belief states from model output (new schema)."""

    @staticmethod
    def parse_belief(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and validate <belief>...</belief> JSON.

        Expected format:
        {
          "state": {
            "objects": {"cup 1": "sidetable 1"},
            "states":  {"cup 1": "heated"},
            "visited": ["sidetable 1", "microwave 1"]
          },
          "task": {
            "phase":   "transform",
            "target":  "cup 1",
            "subgoal": "Heat cup 1 in microwave 1"
          },
          "prediction": "If I go to microwave 1, I expect ..."
        }
        """
        m = re.search(r'<belief>(.*?)</belief>', text, re.DOTALL | re.IGNORECASE)
        if not m:
            return None

        belief_text = m.group(1).strip()
        try:
            data = json.loads(belief_text)
        except json.JSONDecodeError:
            # Try with single→double quote normalisation
            try:
                data = json.loads(belief_text.replace("'", '"'))
            except json.JSONDecodeError:
                return None

        if not isinstance(data, dict):
            return None

        # New schema
        if 'state' in data or 'task' in data or 'prediction' in data:
            state = data.get('state', {}) or {}
            task  = data.get('task',  {}) or {}
            return {
                'state': {
                    'objects':             state.get('objects',             {}) or {},
                    'states':              state.get('states',              {}) or {},
                    'visited':             state.get('visited',             []) or [],
                    'unvisited_candidates': state.get('unvisited_candidates', []) or [],
                },
                'task': {
                    'phase':   str(task.get('phase',   '') or '').lower().strip(),
                    'target':  str(task.get('target',  '') or '').lower().strip(),
                    'subgoal': str(task.get('subgoal', '') or '').strip(),
                },
                'prediction': str(data.get('prediction', '') or '').strip(),
            }

        # Legacy schema fallback (world_model_update / task_progress_update)
        if 'world_model_update' in data or 'task_progress_update' in data:
            wm = data.get('world_model_update', {}) or {}
            tp = data.get('task_progress_update', {}) or {}
            return {
                'state': {
                    'objects': wm.get('found_objects', {}) or {},
                    'states':  wm.get('state_changes',  {}) or {},
                    'visited': wm.get('cleared_receptacles', []) or [],
                },
                'task': {
                    'phase':   str(tp.get('subgoal_status', '') or '').lower().strip(),
                    'target':  '',
                    'subgoal': str(tp.get('updated_subgoal', '') or '').strip(),
                },
                'prediction': '',
            }

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth tracker (from observation text)
# ─────────────────────────────────────────────────────────────────────────────

class GroundTruthTracker:
    """Accumulate verified world facts by parsing ALFWorld observation text."""

    def __init__(self):
        self.visible_receptacles: set = set()
        self.visited_locations: set   = set()
        self.object_locations: Dict[str, str] = {}
        self.object_states:    Dict[str, str] = {}
        self.cleared_receptacles: set = set()
        self.current_inventory: Optional[str] = None
        self.interaction_history: List[str]   = []
        self.task_goal: Optional[str] = None

    def update_from_observation(self, obs: str, action: str = None):
        obs_lower = obs.lower()

        # Initial room scan — receptacles visible but not yet checked
        m = re.search(
            r'you are in the middle of a room\. looking quickly around you, you see (.*?)\.',
            obs_lower
        )
        if m:
            desc = m.group(1).replace(', and ', ', ')
            for recep in re.findall(r'(?:a |an )([\w\s]+\d+)', desc):
                self.visible_receptacles.add(recep.strip())

        # Facing a receptacle (visible, not necessarily checked)
        m2 = re.search(r'you are facing (.*?)\.', obs_lower)
        if m2:
            self.visible_receptacles.add(m2.group(1).strip())

        # Pick-up → inventory
        m3 = re.search(r'you (?:pick up|take) ([\w\s]+\d+)', obs_lower)
        if m3:
            self.current_inventory = m3.group(1).strip()

        # Put → object placed
        m4 = re.search(r'you put ([\w\s]+\d+) (?:in|on) ([\w\s]+\d+)', obs_lower)
        if m4:
            obj, loc = m4.group(1).strip(), m4.group(2).strip()
            self.object_locations[obj] = f'in/on {loc}'
            self.current_inventory = None

        # "On the X, you see Y" — visited + contents
        for loc, items_str in re.findall(r'on the ([\w\s]+\d+), you see (.*?)\.', obs_lower):
            loc = loc.strip()
            self.visited_locations.add(loc)
            self.visible_receptacles.add(loc)
            self._parse_items(items_str, f'on {loc}', loc)

        # "In the X, you see Y" — visited + contents
        for loc, items_str in re.findall(r'in the ([\w\s]+\d+), you see (.*?)\.', obs_lower):
            loc = loc.strip()
            self.visited_locations.add(loc)
            self.visible_receptacles.add(loc)
            self._parse_items(items_str, f'in {loc}', loc)

        # Object states: "the X is open/closed/clean/dirty/hot/cold"
        for obj, state in re.findall(
            r'the ([\w\s]+\d+) is (open|closed|clean|dirty|hot|cold)', obs_lower
        ):
            obj = obj.strip()
            self.object_states[obj] = state
            if state == 'open':
                self.visited_locations.add(obj)
                self.visible_receptacles.add(obj)

        # Transform verbs → states
        for pattern, state in [
            (r'you heat ([\w\s]+\d+)',  'hot'),
            (r'you clean ([\w\s]+\d+)', 'clean'),
            (r'you cool ([\w\s]+\d+)',  'cold'),
        ]:
            m5 = re.search(pattern, obs_lower)
            if m5:
                self.object_states[m5.group(1).strip()] = state

        if action:
            self.interaction_history.append(action.lower())

        if self.task_goal is None:
            m6 = re.search(r'your task is to: (.*?)\.', obs_lower)
            if m6:
                self.task_goal = m6.group(1).strip()

    def _parse_items(self, items_str: str, location: str, loc_key: str):
        if 'nothing' in items_str:
            self.cleared_receptacles.add(loc_key)
            return
        items_str = items_str.replace(', and ', ', ').replace(' and ', ', ')
        for item in re.findall(r'(?:a |an )([\w\s]+\d+)', items_str):
            item = item.strip()
            if item:
                self.object_locations[item] = location

    def get_ground_truth_state(self) -> Dict[str, Any]:
        unvisited = sorted(self.visible_receptacles - self.visited_locations)
        return {
            'visible_receptacles':   list(self.visible_receptacles),
            'visited':               list(self.visited_locations),
            'unvisited_candidates':  unvisited,
            'object_locations':      dict(self.object_locations),
            'object_states':         dict(self.object_states),
            'cleared_receptacles':   list(self.cleared_receptacles),
            'current_inventory':     self.current_inventory,
            'interactions':          self.interaction_history.copy(),
            'task_goal':             self.task_goal,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reward calculator
# ─────────────────────────────────────────────────────────────────────────────

class RebelRewardCalculator:
    """
    Compute r_state, r_task, r_pred, r_format for one step.

    r_state : belief.state vs GroundTruthTracker (accumulated observation facts)
    r_task  : belief.task.phase vs env GT phase (admissible_commands)
    r_pred  : belief.prediction keywords vs o_{t+1}
    r_format: output structure compliance
    """

    def __init__(self, alpha=0.3, beta=0.4, gamma=0.3, delta=0.1,
                 use_belief_reward=True,
                 belief_reward_decay_enable=False,
                 belief_reward_decay_method='cosine',
                 belief_reward_warmup_epochs=5,
                 belief_reward_decay_start_epoch=10,
                 belief_reward_decay_end_epoch=60,
                 belief_reward_min_weight=0.1,
                 belief_reward_adaptive_decay=False,
                 belief_reward_target_sr=0.90,
                 belief_reward_decay_alpha=2.0,
                 belief_reward_progress_decay_rate=0.7,
                 belief_reward_consistency_decay_rate=1.0,
                 belief_reward_exploration_decay_rate=2.0):
        # alpha=r_state, beta=r_task, gamma=r_pred, delta=r_format
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.use_belief_reward = use_belief_reward

        # Decay schedule (unchanged from V9/V11)
        self.belief_reward_decay_enable     = belief_reward_decay_enable
        self.belief_reward_decay_method     = belief_reward_decay_method
        self.belief_reward_warmup_epochs    = belief_reward_warmup_epochs
        self.belief_reward_decay_start_epoch = belief_reward_decay_start_epoch
        self.belief_reward_decay_end_epoch  = belief_reward_decay_end_epoch
        self.belief_reward_min_weight       = belief_reward_min_weight
        self.current_epoch = 0

        self.belief_reward_adaptive_decay = belief_reward_adaptive_decay
        self.belief_reward_target_sr      = belief_reward_target_sr
        self.belief_reward_decay_alpha    = belief_reward_decay_alpha
        self.current_success_rate         = None

        self.progress_decay_rate     = belief_reward_progress_decay_rate
        self.consistency_decay_rate  = belief_reward_consistency_decay_rate
        self.exploration_decay_rate  = belief_reward_exploration_decay_rate

        self.format_valid_reward    =  0.01
        self.format_invalid_penalty = -0.05
        self.action_invalid_penalty = -0.02

    # ------------------------------------------------------------------
    # Decay schedule (unchanged)
    # ------------------------------------------------------------------

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def set_success_rate(self, success_rate: float):
        self.current_success_rate = success_rate

    def get_belief_reward_weight(self) -> float:
        import math
        if not self.belief_reward_decay_enable:
            return 1.0 if self.use_belief_reward else 0.0

        epoch      = self.current_epoch
        warmup     = self.belief_reward_warmup_epochs
        decay_start = self.belief_reward_decay_start_epoch
        decay_end  = self.belief_reward_decay_end_epoch
        min_weight = self.belief_reward_min_weight

        if epoch < warmup:
            return epoch / warmup if warmup > 0 else 1.0

        base_weight = 1.0

        if self.belief_reward_adaptive_decay and self.current_success_rate is not None:
            sr         = self.current_success_rate
            target_sr  = self.belief_reward_target_sr
            alpha      = self.belief_reward_decay_alpha
            if target_sr > 0:
                decay_factor = max(min_weight, 1.0 - (sr / target_sr) ** alpha)
            else:
                decay_factor = 1.0
            base_weight *= decay_factor

        if not self.belief_reward_adaptive_decay and epoch < decay_start:
            return 1.0

        if epoch > decay_start:
            progress = min(1.0, max(0.0, (epoch - decay_start) / (decay_end - decay_start)))
            if self.belief_reward_decay_method == 'linear':
                cw = 1.0 - progress * (1.0 - min_weight)
            elif self.belief_reward_decay_method == 'exponential':
                cw = min_weight + (1.0 - min_weight) * math.exp(-3 * progress)
            else:  # cosine (default)
                cw = min_weight + (1.0 - min_weight) * 0.5 * (1 + math.cos(math.pi * progress))
            cw = max(min_weight, cw)
            base_weight = min(base_weight, cw) if self.belief_reward_adaptive_decay else cw

        return max(min_weight, base_weight)

    def compute_component_weights(self, base_weight: float) -> Dict[str, float]:
        if base_weight <= 0:
            return {'state': 0.0, 'task': 0.0, 'pred': 0.0, 'format': 1.0}
        bw = max(1e-8, min(1.0, base_weight))
        return {
            'state':  bw ** self.consistency_decay_rate,
            'task':   bw ** self.progress_decay_rate,
            'pred':   bw ** self.progress_decay_rate,
            'format': 1.0,
        }

    # ------------------------------------------------------------------
    # r_state: belief.state vs ground truth (cumulative observation facts)
    # ------------------------------------------------------------------

    def calculate_state_reward(
        self,
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int = -1,
    ) -> float:
        """
        Compare belief.state.{objects, states, visited, unvisited_candidates}
        with GroundTruthTracker using conflict-based scoring.

        objects / states / unvisited_candidates: only penalize DIRECT CONTRADICTIONS
          (LLM asserts X=A but GT confirms X=B) and GT-confirmed facts missed by LLM.
          LLM inferences about unvisited locations are NOT penalized.
        visited: F1 against GT (agent's own movement history — no inference involved).

        Weights: objects 35%, states 25%, visited 25%, unvisited_candidates 15%.
        """
        state = belief_state.get('state', {}) or {}
        gt_objects   = ground_truth.get('object_locations', {}) or {}
        gt_states    = ground_truth.get('object_states', {})   or {}
        gt_visited   = set(v.lower() for v in (ground_truth.get('visited', []) or []))
        gt_unvisited = set(v.lower() for v in (ground_truth.get('unvisited_candidates', []) or []))

        components = []

        use_lenient = (step >= 0 and step < 2)

        # 1. objects — conflict-based (40%)
        belief_objects = state.get('objects', {}) or {}
        if not isinstance(belief_objects, dict):
            belief_objects = {}

        if use_lenient:
            components.append(0.6 if belief_objects else 0.4)
        elif belief_objects:
            # "in_hand" checked against inventory (still direct-verifiable)
            contradictions, gt_misses = 0, 0
            for obj, loc in belief_objects.items():
                if not isinstance(loc, str):
                    continue
                if 'in_hand' in loc.lower():
                    inv = ground_truth.get('current_inventory', '') or ''
                    if obj.lower() not in inv.lower():
                        contradictions += 1
                    continue
                # Check for direct contradiction: GT knows where obj is, LLM disagrees
                for gt_obj, gt_loc in gt_objects.items():
                    if not isinstance(gt_loc, str):
                        continue
                    if obj.lower() in gt_obj.lower() or gt_obj.lower() in obj.lower():
                        if loc.lower() not in gt_loc.lower() and gt_loc.lower() not in loc.lower():
                            contradictions += 1
                        break
                # LLM inference about unvisited location → not penalized (no else branch)
            # GT-confirmed facts missed by LLM
            for gt_obj in gt_objects:
                matched = any(
                    gt_obj.lower() in obj.lower() or obj.lower() in gt_obj.lower()
                    for obj in belief_objects
                )
                if not matched:
                    gt_misses += 1
            gt_confirmed = len(gt_objects)
            if gt_confirmed > 0:
                components.append(max(0.0, 1.0 - (contradictions + gt_misses) / gt_confirmed))
            else:
                components.append(0.7)  # GT knows nothing yet → LLM inference unchecked
        else:
            # No belief objects at all
            components.append(0.0 if gt_objects else 0.5)

        # 2. object states — conflict-based (30%)
        belief_states = state.get('states', {}) or {}
        if not isinstance(belief_states, dict):
            belief_states = {}

        if use_lenient:
            components.append(0.6 if belief_states else 0.5)
        elif belief_states:
            contradictions, gt_misses = 0, 0
            for obj, st in belief_states.items():
                if not isinstance(st, str):
                    continue
                for gt_obj, gt_st in gt_states.items():
                    if not isinstance(gt_st, str):
                        continue
                    if obj.lower() in gt_obj.lower() or gt_obj.lower() in obj.lower():
                        if st.lower() != gt_st.lower():
                            contradictions += 1
                        break
                # States for unvisited objects → not penalized
            for gt_obj in gt_states:
                matched = any(
                    gt_obj.lower() in obj.lower() or obj.lower() in gt_obj.lower()
                    for obj in belief_states
                )
                if not matched:
                    gt_misses += 1
            gt_confirmed = len(gt_states)
            if gt_confirmed > 0:
                components.append(max(0.0, 1.0 - (contradictions + gt_misses) / gt_confirmed))
            else:
                components.append(0.7)
        else:
            components.append(0.0 if gt_states else 0.5)

        # 3. visited — F1 (agent's own movement, GT is ground truth, no inference) (30%)
        belief_visited = state.get('visited', []) or []
        if not isinstance(belief_visited, list):
            belief_visited = []
        belief_visited_set = set(v.lower() for v in belief_visited if isinstance(v, str))

        if use_lenient:
            components.append(0.6 if belief_visited_set else 0.5)
        elif belief_visited_set and gt_visited:
            tp   = len(belief_visited_set & gt_visited)
            fp   = len(belief_visited_set - gt_visited)
            prec = tp / len(belief_visited_set)
            rec  = tp / len(gt_visited)
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            components.append(max(0.0, f1 - fp * 0.1))
        elif not belief_visited_set and not gt_visited:
            components.append(0.6)
        else:
            components.append(0.3)

        # 4. unvisited_candidates — conflict-based (15%)
        # Penalize only when GT confirmed a location is fully explored but LLM still lists it.
        belief_unvisited = state.get('unvisited_candidates', []) or []
        if not isinstance(belief_unvisited, list):
            belief_unvisited = []
        belief_unvisited_set = set(v.lower() for v in belief_unvisited if isinstance(v, str))

        if use_lenient:
            components.append(0.6 if belief_unvisited_set else 0.5)
        elif belief_unvisited_set:
            # Direct contradiction: LLM lists a location as unvisited but GT says it's visited
            false_unvisited = len(belief_unvisited_set & gt_visited)
            if belief_unvisited_set:
                penalty = false_unvisited / len(belief_unvisited_set)
                components.append(max(0.0, 1.0 - penalty))
            else:
                components.append(0.7)
        elif not belief_unvisited_set and not gt_unvisited:
            components.append(0.8)
        else:
            components.append(0.4)

        return 0.35 * components[0] + 0.25 * components[1] + 0.25 * components[2] + 0.15 * components[3]

    # ------------------------------------------------------------------
    # r_task: belief.task.phase vs env GT (admissible_commands)
    # ------------------------------------------------------------------

    def calculate_task_reward(
        self,
        belief_state: Dict[str, Any],
        admissible_commands: List[str],
        done: bool = False,
        success: bool = False,
    ) -> float:
        """
        Hard supervision: compare belief.task.phase with extract_phase_gt_alfworld().
        Returns 1.0 on match, 0.0 on mismatch (binary signal).
        """
        task = belief_state.get('task', {}) or {}
        model_phase = str(task.get('phase', '') or '').lower().strip()

        if done and success:
            return 1.0 if model_phase == 'done' else 0.0

        holding = False
        state = belief_state.get('state', {}) or {}
        objects = state.get('objects', {}) or {}
        if isinstance(objects, dict):
            holding = any('in_hand' in str(v).lower() for v in objects.values())

        gt_phase = extract_phase_gt_alfworld(admissible_commands, holding=holding)
        return 1.0 if model_phase == gt_phase else 0.0

    # ------------------------------------------------------------------
    # r_pred: prediction keyword match vs next observation
    # ------------------------------------------------------------------

    def calculate_pred_reward(
        self,
        belief_state: Dict[str, Any],
        next_obs: Optional[str],
    ) -> float:
        """Keyword match between belief.prediction and o_{t+1}."""
        if not next_obs:
            return 0.0
        prediction = str(belief_state.get('prediction', '') or '').strip()
        return compute_r_pred(prediction, next_obs)

    # ------------------------------------------------------------------
    # r_format
    # ------------------------------------------------------------------

    def calculate_format_reward(
        self,
        is_format_valid: bool,
        is_action_available: bool,
    ) -> float:
        r = self.format_valid_reward if is_format_valid else self.format_invalid_penalty
        if is_format_valid and not is_action_available:
            r += self.action_invalid_penalty
        return r

    # ------------------------------------------------------------------
    # Total intrinsic reward
    # ------------------------------------------------------------------

    def calculate_total_intrinsic_reward(
        self,
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int,
        done: bool,
        success: bool,
        is_format_valid: bool = True,
        is_action_available: bool = True,
        admissible_commands: Optional[List[str]] = None,
        next_obs: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        R_cons = α*r_state + β*r_task + γ*r_pred  (+ δ*r_format independently)

        Args:
            belief_state:        parsed belief from current step
            ground_truth:        GroundTruthTracker.get_ground_truth_state()
            step:                current step index
            done / success:      episode terminal flags
            admissible_commands: from env info, used for r_task
            next_obs:            o_{t+1} text, used for r_pred (None on last step)
            task_type:           optional, for logging
        """
        r_state  = self.calculate_state_reward(belief_state, ground_truth, step)
        r_task   = self.calculate_task_reward(belief_state, admissible_commands or [], done, success)
        r_pred   = self.calculate_pred_reward(belief_state, next_obs)
        r_format = self.calculate_format_reward(is_format_valid, is_action_available)

        belief_weight     = self.get_belief_reward_weight()
        if not self.use_belief_reward and not self.belief_reward_decay_enable:
            belief_weight = 0.0

        cw = self.compute_component_weights(belief_weight)

        total_reward = (
            self.alpha * r_state  * cw['state']  +
            self.beta  * r_task   * cw['task']   +
            self.gamma * r_pred   * cw['pred']   +
            self.delta * r_format * cw['format']
        )

        return total_reward, {
            'r_state':   r_state,
            'r_task':    r_task,
            'r_pred':    r_pred,
            'r_format':  r_format,
            'r_intrinsic_total': total_reward,
            'belief_weight': belief_weight,
            'r_state_weighted':  r_state  * cw['state'],
            'r_task_weighted':   r_task   * cw['task'],
            'r_pred_weighted':   r_pred   * cw['pred'],
            'current_epoch': self.current_epoch,
            'component_weight_state':  cw['state'],
            'component_weight_task':   cw['task'],
            'component_weight_pred':   cw['pred'],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Deviation metrics (for analysis / paper experiments)
# ─────────────────────────────────────────────────────────────────────────────

class BeliefDeviationCalculator:
    """Compute alignment metrics between belief.state and ground truth."""

    @staticmethod
    def compute_object_location_deviation(
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, float]:
        state = belief_state.get('state', {}) or {}
        predicted = state.get('objects', {}) or {}
        actual    = ground_truth.get('object_locations', {}) or {}

        if not predicted and not actual:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'deviation': 0.0}
        if not predicted:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'deviation': 1.0}
        if not actual:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'deviation': 1.0}

        pred_n = {k.lower().strip(): str(v).lower().strip() for k, v in predicted.items() if k and v}
        act_n  = {k.lower().strip(): str(v).lower().strip() for k, v in actual.items()    if k and v}

        tp = sum(
            1 for obj, loc in pred_n.items()
            for a_obj, a_loc in act_n.items()
            if (obj in a_obj or a_obj in obj) and (loc in a_loc or a_loc in loc)
        )
        prec = tp / len(pred_n) if pred_n else 0.0
        rec  = tp / len(act_n)  if act_n  else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {'precision': prec, 'recall': rec, 'f1': f1, 'deviation': 1.0 - f1}

    @staticmethod
    def compute_state_deviation(
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, float]:
        state = belief_state.get('state', {}) or {}
        predicted = state.get('states', {}) or {}
        actual    = ground_truth.get('object_states', {}) or {}

        if not predicted and not actual:
            return {'accuracy': 1.0, 'deviation': 0.0, 'n_predictions': 0}
        if not predicted:
            return {'accuracy': 0.0, 'deviation': 1.0, 'n_predictions': 0}

        correct, total = 0, 0
        for obj, pred_st in predicted.items():
            if not isinstance(pred_st, str):
                continue
            total += 1
            for a_obj, a_st in actual.items():
                if isinstance(a_st, str) and (obj.lower() in a_obj.lower() or a_obj.lower() in obj.lower()):
                    if pred_st.lower().strip() == a_st.lower().strip():
                        correct += 1
                    break

        acc = correct / total if total > 0 else 0.0
        return {'accuracy': acc, 'deviation': 1.0 - acc, 'n_predictions': total}

    @staticmethod
    def compute_exploration_deviation(
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, float]:
        state = belief_state.get('state', {}) or {}
        pred_v  = set(v.lower() for v in (state.get('visited', []) or [])  if isinstance(v, str))
        act_v   = set(v.lower() for v in (ground_truth.get('visited', []) or []) if isinstance(v, str))

        if not pred_v and not act_v:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'deviation': 0.0}
        if not pred_v:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'deviation': 1.0}
        if not act_v:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'deviation': 1.0}

        tp   = len(pred_v & act_v)
        prec = tp / len(pred_v)
        rec  = tp / len(act_v)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {'precision': prec, 'recall': rec, 'f1': f1, 'deviation': 1.0 - f1}

    @staticmethod
    def compute_total_belief_deviation(
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        weights = weights or {'object_location': 0.4, 'state': 0.3, 'exploration': 0.3}

        if not belief_state or not isinstance(belief_state, dict):
            return {
                'object_location': {'deviation': 1.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'state':           {'deviation': 1.0, 'accuracy': 0.0, 'n_predictions': 0},
                'exploration':     {'deviation': 1.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'total_deviation': 1.0,
                'belief_valid': False,
            }

        ol = BeliefDeviationCalculator.compute_object_location_deviation(belief_state, ground_truth)
        st = BeliefDeviationCalculator.compute_state_deviation(belief_state, ground_truth)
        ex = BeliefDeviationCalculator.compute_exploration_deviation(belief_state, ground_truth)

        total = (
            weights['object_location'] * ol['deviation'] +
            weights['state']           * st['deviation'] +
            weights['exploration']     * ex['deviation']
        )
        return {
            'object_location': ol,
            'state':           st,
            'exploration':     ex,
            'total_deviation': total,
            'belief_valid':    True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_rebel_tracker():
    return BeliefStateParser(), RebelRewardCalculator()

def create_belief_deviation_calculator():
    return BeliefDeviationCalculator()
