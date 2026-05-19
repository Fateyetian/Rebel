"""
WebShop ReBel Belief Tracking and Intrinsic Reward

Three reward signals:
  r_state : belief.state.{target, confirmed, unconfirmed} vs GT (page content)
  r_task  : belief.task.phase vs env GT phase (page_type from observation)
  r_pred  : belief.prediction keyword match vs o_{t+1}
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# GT phase extraction from WebShop observation
# ─────────────────────────────────────────────────────────────────────────────

def extract_phase_gt_webshop(obs: str) -> str:
    """
    Derive ground-truth task phase from WebShop observation text.

    Phase ladder: searching → browsing → viewing → selecting → buying → done
    """
    obs_lower = obs.lower()

    if 'thank you' in obs_lower or 'your score' in obs_lower:
        return 'done'

    if 'buy now' in obs_lower:
        # Selecting: product page WITH clickable option buttons
        if re.search(r'\b(size|color|style|material)\b', obs_lower):
            return 'selecting'
        return 'viewing'

    if 'back to search' in obs_lower:
        if 'results for' in obs_lower or re.search(r'\bb\d{9,}\b', obs_lower):
            return 'browsing'
        return 'browsing'

    return 'searching'


# ─────────────────────────────────────────────────────────────────────────────
# r_pred utility (shared with ALFWorld)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    'that', 'will', 'the', 'and', 'or', 'to', 'a', 'an', 'in', 'on', 'at',
    'be', 'see', 'find', 'expect', 'next', 'observation', 'if', 'i', 'you',
    'then', 'with', 'from', 'this', 'it', 'is', 'are', 'was', 'not', 'no',
}

_HIGH_NOISE_UNIGRAMS = {
    'closed', 'open', 'opened', 'pick', 'picked', 'found', 'clean', 'cleaned',
    'heat', 'heated', 'cool', 'cooled', 'put', 'place', 'placed', 'hold',
    'holding', 'examine', 'look', 'move', 'moved', 'take', 'taken',
    'search', 'click', 'back', 'item', 'page', 'product',
}


def compute_r_pred(prediction: str, next_obs: str) -> float:
    """Bigram-anchored keyword match between belief.prediction and o_{t+1}."""
    if not prediction or not next_obs:
        return 0.0

    pred_lower = prediction.lower()
    obs_lower  = next_obs.lower()

    m = re.search(r'i expect\s+(.*)', pred_lower)
    expected = m.group(1) if m else pred_lower

    tokens = [
        w for w in re.findall(r'\b[a-z]\w+\b', expected)
        if len(w) > 3 and w not in _STOPWORDS
    ]
    if not tokens:
        return 0.0

    features: list = []
    for i, tok in enumerate(tokens):
        if tok not in _HIGH_NOISE_UNIGRAMS:
            features.append(tok)
        if i < len(tokens) - 1:
            features.append(f'{tok} {tokens[i + 1]}')

    if not features:
        features = tokens

    return sum(1 for feat in features if feat in obs_lower) / len(features)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class WebShopBeliefStateParser:
    """Parse belief states from model output (new WebShop schema)."""

    @staticmethod
    def parse_belief(text: str) -> Optional[Dict[str, Any]]:
        """
        Expected format:
        {
          "state": {
            "target":      {"color": "blue", "size": "8"},
            "confirmed":   {"color": "blue"},
            "unconfirmed": ["size"],
            "product_id":  "B07XYZ123"
          },
          "task": {
            "phase":   "selecting",
            "target":  "B07XYZ123",
            "subgoal": "Select size 8 from options"
          },
          "prediction": "If I click[8], I expect ..."
        }
        """
        m = re.search(r'<belief>(.*?)</belief>', text, re.DOTALL | re.IGNORECASE)
        if not m:
            return None

        belief_text = m.group(1).strip()
        try:
            data = json.loads(belief_text)
        except json.JSONDecodeError:
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
                    'target':      state.get('target',      {}) or {},
                    'confirmed':   state.get('confirmed',   {}) or {},
                    'unconfirmed': state.get('unconfirmed', []) or [],
                    'product_id':  str(state.get('product_id', '') or '').strip(),
                },
                'task': {
                    'phase':   str(task.get('phase',   '') or '').lower().strip(),
                    'target':  str(task.get('target',  '') or '').strip(),
                    'subgoal': str(task.get('subgoal', '') or '').strip(),
                },
                'prediction': str(data.get('prediction', '') or '').strip(),
            }

        # Legacy fallback (product_understanding / search_progress / exploration_state)
        if 'search_progress' in data or 'product_understanding' in data:
            pu = data.get('product_understanding', {}) or {}
            sp = data.get('search_progress',       {}) or {}
            av = data.get('attribute_verification',{}) or {}
            return {
                'state': {
                    'target':      pu.get('target_attributes', {}),
                    'confirmed':   {v['attribute']: v['value']
                                    for v in (av.get('verified', []) or [])
                                    if isinstance(v, dict) and 'attribute' in v and 'value' in v},
                    'unconfirmed': av.get('unverified', []) or [],
                    'product_id':  '',
                },
                'task': {
                    'phase':   str(sp.get('search_status', '') or '').lower().strip(),
                    'target':  '',
                    'subgoal': str(sp.get('updated_subgoal', '') or '').strip(),
                },
                'prediction': '',
            }

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth tracker
# ─────────────────────────────────────────────────────────────────────────────

class WebShopGroundTruthTracker:
    """Track ground truth state by parsing WebShop observations and actions."""

    def __init__(self):
        self.page_type: str = 'searching'
        self.products_seen: List[str]   = []
        self.options_clicked: List[str] = []
        self.search_queries: List[str]  = []
        self.current_product_title:  Optional[str] = None
        self.current_product_price:  Optional[str] = None
        self.current_product_attrs:  Dict[str, str] = {}
        self.available_clickables:   List[str] = []
        self.task_goal: Optional[str] = None
        self.interaction_history: List[str] = []

    def update_from_observation(self, obs: str, action: str = None, info: dict = None):
        obs_lower = obs.lower()

        # Page type from observation structure
        self.page_type = extract_phase_gt_webshop(obs)

        # Available actions from info
        if info and 'available_actions' in info:
            avail = info['available_actions']
            self.available_clickables = avail.get('clickables', []) if isinstance(avail, dict) else []

        # Track actions
        if action:
            action_lower = action.lower().strip()
            self.interaction_history.append(action_lower)

            m_search = re.match(r'search\[(.+)\]', action_lower)
            if m_search:
                q = m_search.group(1).strip()
                if q not in self.search_queries:
                    self.search_queries.append(q)

            m_click = re.match(r'click\[(.+)\]', action_lower)
            if m_click:
                clicked = m_click.group(1).strip()
                if re.match(r'^b\d+', clicked):
                    if clicked not in self.products_seen:
                        self.products_seen.append(clicked)
                elif clicked not in ('buy now', 'back to search', '< prev', 'next >'):
                    if clicked not in self.options_clicked:
                        self.options_clicked.append(clicked)

        self._parse_product_info(obs)

    def _parse_product_info(self, obs: str):
        # Price
        m = re.search(r'\$([\d.]+)', obs)
        if m:
            self.current_product_price = f'${m.group(1)}'

        # Title: longest non-nav text segment
        parts = obs.split(' [SEP] ')
        for part in parts:
            p = part.strip()
            if (len(p) > 20 and '$' not in p
                    and p.lower() not in ('back to search', 'buy now', 'search',
                                          '< prev', 'next >')):
                self.current_product_title = p
                break

        # Option values for confirmed attributes
        for m in re.finditer(r'\b(size|color|style|material)\b[:\s]+([^\[,\n]+)', obs.lower()):
            self.current_product_attrs[m.group(1).strip()] = m.group(2).strip()

    def get_ground_truth_state(self) -> Dict[str, Any]:
        return {
            'page_type':             self.page_type,
            'products_seen':         self.products_seen.copy(),
            'options_clicked':       self.options_clicked.copy(),
            'search_queries':        self.search_queries.copy(),
            'current_product_title': self.current_product_title,
            'current_product_price': self.current_product_price,
            'current_product_attrs': dict(self.current_product_attrs),
            'available_clickables':  self.available_clickables.copy(),
            'task_goal':             self.task_goal,
            'interactions':          self.interaction_history.copy(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reward calculator
# ─────────────────────────────────────────────────────────────────────────────

class WebShopRebelRewardCalculator:
    """
    r_state : belief.state.{target, confirmed, unconfirmed} vs GT page content
    r_task  : belief.task.phase vs extract_phase_gt_webshop(obs)
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
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.use_belief_reward = use_belief_reward

        self.belief_reward_decay_enable      = belief_reward_decay_enable
        self.belief_reward_decay_method      = belief_reward_decay_method
        self.belief_reward_warmup_epochs     = belief_reward_warmup_epochs
        self.belief_reward_decay_start_epoch = belief_reward_decay_start_epoch
        self.belief_reward_decay_end_epoch   = belief_reward_decay_end_epoch
        self.belief_reward_min_weight        = belief_reward_min_weight
        self.current_epoch                   = 0

        self.belief_reward_adaptive_decay = belief_reward_adaptive_decay
        self.belief_reward_target_sr      = belief_reward_target_sr
        self.belief_reward_decay_alpha    = belief_reward_decay_alpha
        self.current_success_rate         = None

        self.progress_decay_rate    = belief_reward_progress_decay_rate
        self.consistency_decay_rate = belief_reward_consistency_decay_rate
        self.exploration_decay_rate = belief_reward_exploration_decay_rate

        self.format_valid_reward    =  0.01
        self.format_invalid_penalty = -0.05
        self.action_invalid_penalty = -0.02

    # ------------------------------------------------------------------
    # Decay schedule (same as ALFWorld)
    # ------------------------------------------------------------------

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def set_success_rate(self, success_rate: float):
        self.current_success_rate = success_rate

    def get_belief_reward_weight(self) -> float:
        import math
        if not self.belief_reward_decay_enable:
            return 1.0 if self.use_belief_reward else 0.0

        epoch       = self.current_epoch
        warmup      = self.belief_reward_warmup_epochs
        decay_start = self.belief_reward_decay_start_epoch
        decay_end   = self.belief_reward_decay_end_epoch
        min_weight  = self.belief_reward_min_weight

        if epoch < warmup:
            return epoch / warmup if warmup > 0 else 1.0

        base_weight = 1.0
        if self.belief_reward_adaptive_decay and self.current_success_rate is not None:
            sr = self.current_success_rate
            if self.belief_reward_target_sr > 0:
                decay_factor = max(min_weight, 1.0 - (sr / self.belief_reward_target_sr) ** self.belief_reward_decay_alpha)
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
            else:
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
    # r_state: belief.state vs ground truth
    # ------------------------------------------------------------------

    def calculate_state_reward(
        self,
        belief_state: Dict[str, Any],
        ground_truth: Dict[str, Any],
        step: int = -1,
    ) -> float:
        """
        Check:
        1. state.target attributes are grounded in task_goal (40%)
        2. state.confirmed are subset of actually-clicked options (35%)
        3. buying-gate: task.phase == 'buying' only when state.unconfirmed is empty (25%)
        """
        state = belief_state.get('state', {}) or {}
        task  = belief_state.get('task',  {}) or {}
        components = []

        # 1. target attributes grounded in task_goal
        target = state.get('target', {}) or {}
        task_goal = str(ground_truth.get('task_goal', '') or '').lower()
        if isinstance(target, dict) and target:
            task_keywords = set(w for w in re.findall(r'\b\w+\b', task_goal) if len(w) > 2)
            matches = sum(
                1 for attr, val in target.items()
                if str(attr).lower() in task_keywords or str(val).lower() in task_keywords
            )
            components.append(min(1.0, 0.4 + 0.6 * matches / len(target)))
        else:
            components.append(0.3)

        # 2. confirmed — discriminative evidence scoring (35%)
        # Score each confirmed attribute by how well it is supported by page evidence:
        #   - Direct contradiction with page attrs          → 0.0
        #   - Supported by page evidence (attrs/clickables) → 1.0
        #   - Relevant to task_goal but unverifiable        → 0.6
        #   - On product page but no evidence at all        → 0.2 (mild hallucination)
        #   - On search/browse page (reasonable inference)  → 0.7
        # This keeps the signal discriminative: heavily confirmed = higher score only
        # when confirmed attributes are actually evidenced by the current page.
        confirmed = state.get('confirmed', {}) or {}
        actual_options  = set(o.lower() for o in (ground_truth.get('options_clicked', []) or []))
        page_attrs      = {k.lower(): str(v).lower() for k, v in
                           (ground_truth.get('current_product_attrs', {}) or {}).items()}
        clickables      = set(o.lower() for o in (ground_truth.get('available_clickables', []) or []))
        task_goal_kw    = set(w for w in re.findall(r'\b\w{3,}\b', task_goal))
        page_evidence   = actual_options | clickables | set(page_attrs.keys()) | set(page_attrs.values())
        page_type = ground_truth.get('page_type', 'searching')

        if isinstance(confirmed, dict) and confirmed:
            attr_scores = []
            for attr, val in confirmed.items():
                attr_l, val_l = str(attr).lower(), str(val).lower()
                if attr_l in page_attrs and val_l not in page_attrs[attr_l] and page_attrs[attr_l] not in val_l:
                    # Direct contradiction
                    attr_scores.append(0.0)
                elif val_l in page_evidence or attr_l in page_evidence:
                    # Supported by current page evidence
                    attr_scores.append(1.0)
                elif val_l in task_goal_kw or attr_l in task_goal_kw:
                    # Task-relevant but not yet on-page verified
                    attr_scores.append(0.6)
                elif page_type in ('viewing', 'selecting', 'buying'):
                    # On product page with no evidence: mild hallucination
                    attr_scores.append(0.2)
                else:
                    # Search/browse phase: unverified inference is acceptable
                    attr_scores.append(0.7)
            components.append(sum(attr_scores) / len(attr_scores))
        else:
            components.append(0.4 if page_type in ('viewing', 'selecting') else 0.7)

        # 3. buying-gate invariant
        unconfirmed = state.get('unconfirmed', []) or []
        model_phase = str(task.get('phase', '') or '').lower().strip()
        if model_phase == 'buying':
            # Must have empty unconfirmed
            gate_ok = len(unconfirmed) == 0
            components.append(1.0 if gate_ok else 0.0)
        else:
            components.append(0.8)  # Not claiming buying — no violation

        return 0.4 * components[0] + 0.35 * components[1] + 0.25 * components[2]

    # ------------------------------------------------------------------
    # r_task: belief.task.phase vs page_type GT
    # ------------------------------------------------------------------

    def calculate_task_reward(
        self,
        belief_state: Dict[str, Any],
        current_obs: str,
        done: bool = False,
        success: bool = False,
    ) -> float:
        """Hard supervision: belief.task.phase vs extract_phase_gt_webshop(obs)."""
        task = belief_state.get('task', {}) or {}
        model_phase = str(task.get('phase', '') or '').lower().strip()

        if done and success:
            return 1.0 if model_phase == 'done' else 0.0

        gt_phase = extract_phase_gt_webshop(current_obs)
        return 1.0 if model_phase == gt_phase else 0.0

    # ------------------------------------------------------------------
    # r_pred
    # ------------------------------------------------------------------

    def calculate_pred_reward(
        self,
        belief_state: Dict[str, Any],
        next_obs: Optional[str],
    ) -> float:
        if not next_obs:
            return 0.0
        prediction = str(belief_state.get('prediction', '') or '').strip()
        return compute_r_pred(prediction, next_obs)

    # ------------------------------------------------------------------
    # r_format
    # ------------------------------------------------------------------

    def calculate_format_reward(self, is_format_valid: bool, is_action_available: bool) -> float:
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
        current_obs: str = '',
        next_obs: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        R_cons = α*r_state + β*r_task + γ*r_pred  (+ δ*r_format)

        Args:
            current_obs: o_t text — used for r_task phase extraction
            next_obs:    o_{t+1} text — used for r_pred (None on last step)
        """
        r_state  = self.calculate_state_reward(belief_state, ground_truth, step)
        r_task   = self.calculate_task_reward(belief_state, current_obs, done, success)
        r_pred   = self.calculate_pred_reward(belief_state, next_obs)
        r_format = self.calculate_format_reward(is_format_valid, is_action_available)

        belief_weight = self.get_belief_reward_weight()
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
            'r_state_weighted': r_state * cw['state'],
            'r_task_weighted':  r_task  * cw['task'],
            'r_pred_weighted':  r_pred  * cw['pred'],
            'current_epoch': self.current_epoch,
            'component_weight_state': cw['state'],
            'component_weight_task':  cw['task'],
            'component_weight_pred':  cw['pred'],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_webshop_rebel_tracker():
    return WebShopBeliefStateParser(), WebShopGroundTruthTracker()

def create_webshop_reward_calculator(**kwargs):
    return WebShopRebelRewardCalculator(**kwargs)
