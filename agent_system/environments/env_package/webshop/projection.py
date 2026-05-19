# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple
import re
import json


def webshop_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-20:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = actions[i][-20:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids


def webshop_projection_rebel(actions: List[str], action_infos: List[dict]) -> Tuple[List[str], List[int], List[str], List[bool]]:
    """
    ReBel projection for <belief>/<action> format in WebShop.
    Validates belief JSON structure (search_progress / product_understanding / exploration_state),
    action format (search[...] / click[...]), and ordering. Returns 4-tuple.
    """
    actions_out = []
    valids = []
    beliefs = []
    action_available = [False] * len(actions)

    for i, output in enumerate(actions):
        valid = 1
        act_str = ""
        belief_text = ""
        belief_ok = True

        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        belief_matches = re.findall(r"<belief>(.*?)</belief>", output, re.DOTALL | re.IGNORECASE)
        if len(belief_matches) != 1:
            belief_ok = False
        else:
            belief_text = belief_matches[0].strip()
            try:
                belief_data = json.loads(belief_text.replace("'", '"'))
                if not ('product_understanding' in belief_data or
                        'search_progress' in belief_data or
                        'exploration_state' in belief_data):
                    belief_ok = False
            except json.JSONDecodeError:
                belief_ok = False
            if not belief_text or len(belief_text) < 10:
                belief_ok = False

        action_matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
        if len(action_matches) != 1:
            valid = 0
        else:
            act_str = action_matches[0].strip().lower()
            is_valid_format = bool(
                re.match(r'^search\[.+\]$', act_str) or
                re.match(r'^click\[.+\]$', act_str)
            )
            if not is_valid_format:
                valid = 0
            if i < len(action_infos) and action_infos[i]:
                avail = action_infos[i]
                clickables = avail.get('clickables', [])
                has_search = avail.get('has_search_bar', False)
                search_match = re.match(r'^search\[(.+)\]$', act_str)
                click_match = re.match(r'^click\[(.+)\]$', act_str)
                if search_match and has_search:
                    action_available[i] = True
                elif click_match:
                    clicked_val = click_match.group(1).strip()
                    if clicked_val.lower().strip() in [c.lower().strip() for c in clickables]:
                        action_available[i] = True

        belief_pos = output.lower().find("<belief>")
        action_pos = output.lower().find("<action>")
        if belief_pos != -1 and action_pos != -1 and belief_pos > action_pos:
            belief_ok = False
        if action_pos == -1:
            valid = 0

        if not belief_ok:
            belief_text = ""
        if not act_str:
            act_str = output.lower()[-30:]

        actions_out.append(act_str)
        valids.append(valid)
        beliefs.append(belief_text)

    return actions_out, valids, beliefs, action_available