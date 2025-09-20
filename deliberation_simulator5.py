"""Self-deliberation Multi-agent Debate Simulator"""
import argparse
import logging
import os
import itertools
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple, Optional, Union
from PIL import Image
import csv
import numpy as np
import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from agent_modules.base_agent import BaseAgent
from agent_modules.cot_agent import CoTAgent
from agent_modules.pot_agent import POTAgent
from agent_modules.search_agent import SearchAgent
from agent_modules.knowledge_agent import KnowledgeAgent

from load_pretrainedl import load_causal_lm, load_entailment_classifier
import re

def extract_confidence(text, default=None):
    # Look for 'Confidence:' (case-insensitive) followed by a float
    m = re.search(r'Confidence\s*:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            # clamp to [0,1] if you expect that interval
            if 0 <= val <= 1:
                return val
            # if value outside range, optionally scale or discard
        except ValueError:
            pass
    return default  # e.g. None or previous confidence
from simulation_utils import (
    TEST_API_MODELS,
    TEST_HF_MODELS,
    sample_input_data,
    extract_ans,
    extract_conf,
    extract_conf_metrics,
    eval_ans,
    softmax,
    get_collective_feedback,
)

MEMORY_PATH = "data/memory/"
PROMPTING_STRATEGY_MAPPING = {
    "cot": CoTAgent,
    "pot": POTAgent,
    # "self-ask": SelfAskAgent,
    "search": SearchAgent,
    "knowledge": KnowledgeAgent,
    # "maieutic": MaieuticAgent,
}
MODEL_ENSEMBLE_CHOICES = tuple(
    [
    #"microsoft/Phi-4-multimodal-instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "google/gemma-3-4b-it",
    "ibm-granite/granite-vision-3.3-2b",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    ]
)


def populate_expert_agents(
    selection: Dict[str, int], model_ensemble: bool = False, default_model: str = "google/gemma-3-4b-it", mix_temperature: bool = False, use_vllm: bool = False
) -> Dict[str, Any]:
    pool = {k: [] for k in selection.keys()}
    pretrained_instances = {}
    for choice in set(MODEL_ENSEMBLE_CHOICES):
        if model_ensemble and choice.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values()):
            # load HF instances
            pretrained_instances.update({choice: load_causal_lm(choice, access_token=os.getenv("HF_TOKEN"), use_vllm=use_vllm)})

    for k, count in selection.items():
        for i in range(count):
            temperature = np.random.rand() + 0.5 if mix_temperature else 1
            if k == "gen":
                # if no expertise, must have model ensemble
                for model in MODEL_ENSEMBLE_CHOICES:
                    pool[k].append(
                        BaseAgent(id=f"gen-agent_{i+1}_{model}", model_type=model, pretrained=pretrained_instances.get(model), model_temperature=temperature)
                    )
            elif model_ensemble and k != "self-ask":
                for model in MODEL_ENSEMBLE_CHOICES:
                    pool[k].append(
                        PROMPTING_STRATEGY_MAPPING[k](
                            id=f"{k}-agent_{i+1}_{model}", model_type=model, pretrained=pretrained_instances.get(model), model_temperature=temperature
                        )
                    )
            else:
                # load the pretrained HF instance for your default_model, if it’s one of the HF‐backed models
                pretrained_inst = None
                if default_model.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values()):
                    pretrained_inst = load_causal_lm(
                        default_model,
                        access_token=os.getenv("HF_TOKEN")
                    )
                pool[k].append(
                    PROMPTING_STRATEGY_MAPPING[k](
                        id=f"{k}-agent_{i+1}_{default_model}",
                        model_type=default_model,
                        pretrained=pretrained_inst,
                        model_temperature=temperature,
                    )
                )
    logging.debug(f"agents populated: {pool}")
    return pool


def populate_general_agents(
    size: int,
    model_name: str = "google/gemma-3-4b-it",
    use_vllm: bool = False
) -> List[BaseAgent]:
    """
    Create `size` many generic vision–language agents, each with
    an ID like `general_<model_alias>_<i>`.
    """
    # derive a short alias from the model name
    alias = model_name.split("/")[-1].replace("-", "_")

    # load the model once
    processor, model = load_causal_lm(
        model_name,
        access_token=os.getenv("HF_TOKEN"),
        use_vllm=use_vllm
    )

    agents: List[BaseAgent] = []
    for i in range(size):
        agent_id = f"general_{alias}_{i+1}"
        agents.append(BaseAgent(
            id=agent_id,
            model_type=model_name,
            pretrained=(processor, model),
            model_temperature=1.0,
        ))
    return agents



def allocate_agent_slots(
    sampled_df: pd.DataFrame, n_slots: int = 12, tau: float = 0.2, use_vllm: bool = False
) -> Dict[str, int]:
    """Allocate n slots of expert agents with different specializations (select the best composition of prompting strategies)
    Current approach:
        Initialize one agent per specialization. On m sampled dev questions, get prediction and confidence.
        Rank agents based on average confidence score adjusted on correctness 
            i=1,...,|skills|; j=1,...,|samples|
            c'_ij =  I(a_ij is "Abstain") * ( 2 * I(a_ij is correct) - 1) * c_ij
        Allocate slots according to agent-wise mean confidence:
            Filter out those with mean confidence below some threshold tau (typically > 0)
            Allocate n slots (roughly) proportional to softmax(C_i)
    """
    # get adjusted confidence
    initialization = {"cot": 1, "pot": 0, "search": 0, "knowledge": 1}
    initial_agents = populate_expert_agents(initialization, use_vllm=use_vllm)
    adjusted_confidence_all = {k: [] for k in initialization.keys()}
    sampled_questions = sampled_df["question"].values.tolist()
    reference_answers = sampled_df["reference_answers"].values.tolist()
    sampled_images = sampled_df["image"].tolist()   
    for j, question in enumerate(sampled_questions):
        image = sampled_images[j]  
        for key, agent_group in initial_agents.items():
            for agent in agent_group:
                prob, res = agent.self_deliberate_with_pretrained_instance(
                    query=question,
                    image=image
                )
 # Or True, depending on desired behavior
                if "Abstain" in res:
                    adjusted_confidence_all[key].append(0.0)
                else:
                    correctness = eval_ans(question, extract_ans(res), reference_answers[j])
                    adjusted_confidence_all[key].append((2 * correctness - 1) * extract_conf(res))
                    if key == "pot":
                        logging.debug(f"pot pair: ({extract_ans(res)}, {reference_answers[j]})->{correctness}")

    logging.debug(f"adjusted_confidence_all: {adjusted_confidence_all}")
    adjusted_confidence_mean = {
        k: float(np.mean(adjusted_confidence_all[k])) for k in initialization.keys() if adjusted_confidence_all[k]
    }
    logging.debug(f"adjusted_confidence_mean: {adjusted_confidence_mean}")

    # agent allocation
    final_allocation = {}
    confidence_filtered = {k: c for k, c in adjusted_confidence_mean.items() if c >= tau}
    logging.debug(f"confidence_filtered: {confidence_filtered} from {adjusted_confidence_mean.items()}")
    if len(confidence_filtered):
        portions = softmax(list(confidence_filtered.values()))
        confidence_softmax = {_k: portions[i] for (i, _k) in enumerate(confidence_filtered.keys())} 
        confidence_sorted = dict(sorted(confidence_softmax.items(), key=lambda item: item[1], reverse=True))
        logging.info(f"{portions}, sorted: {confidence_sorted}")
        sorted_keys = list(confidence_sorted.keys())
        # first allocate proportional to floor(softmax(C_i)), then add remaining slots (if any) to the top-ranked agent
        # edge case: n_slots = 2, then for diversity, initialize the top-2 agents (if over one agent type selected)
        if n_slots == 2 and len(sorted_keys) > 1:
            final_allocation[sorted_keys[0]] = 1
            final_allocation[sorted_keys[1]] = 1
        else:
            for i, k in enumerate(sorted_keys):
                final_allocation.update({k: int(np.floor(portions[i] * n_slots))})
            diff = n_slots - sum(final_allocation.values())
            if diff > 0:
                final_allocation[sorted_keys[0]] += diff

        if sum(final_allocation.values()) != n_slots:
            logging.debug(
                f"slots unmatched: {adjusted_confidence_mean}\n{confidence_filtered}\n{final_allocation}"
            )
            raise ValueError
    else:
        logging.info("All agents produced confidence below threshold. Check input task difficulty or initialization.")
        # in this case, allocate all slots to the most confident agent type
        top_key = max(adjusted_confidence_mean, key=adjusted_confidence_mean.get)
        final_allocation[top_key] = n_slots

    logging.info(f"Final allocation of expert agents (for each model): {final_allocation}")
    return final_allocation


def construct_stances(
    votes: List[Tuple[str, str, str, float, Union[float, None], Dict[str, float]]],
    query: str,
    nli_classifier: Optional[Any],
    nli_tokenizer: Optional[Any],
    filter_abstain: Optional[bool] = True,
    conf_rationales: Optional[List[str]] = None,
) -> List[List[Any]]:
    # dict{ans_class: [mean_verb_confidence, [seq_prob], count, confidence_rationale]}
    raw_votes = votes

    classes = {}
    if filter_abstain:
        filtered = [v for v in votes if "abstain" not in v[2].lower()]
        if filtered:
            votes = filtered
        else:
            # Fallback: everyone abstained, so keep at least one "Abstain"
            logging.warning("All votes were Abstain; falling back to unfiltered votes")
            votes = raw_votes

    for i, (id, model, ans, verb_conf, seq_prob, *_) in enumerate(votes):
        logging.debug(f"construct_stances: {i, id, model, ans, verb_conf, seq_prob}")
        conf_rationale = conf_rationales[i] if conf_rationales else None  # for now, only include in stage 2
        if i == 0:
            classes.update({ans: [verb_conf, [seq_prob], 1, conf_rationale]})
        else:
            merged_answer_class = None
            for unique_class in classes.keys():
                if eval_ans(
                    query,
                    ans,
                    unique_class,
                    method="gpt_cls",  # "nli_cls"
                    nli_classifier=nli_classifier,
                    nli_tokenizer=nli_tokenizer,
                ):
                    # not a new class, merge with the equivalent answer class
                    merged_answer_class = unique_class
                    prev_verb_conf, prev_seq_prob, prev_count, prev_rationale = classes[merged_answer_class]
                    prev_seq_prob.append(seq_prob)
                    classes[merged_answer_class] = [
                        (prev_verb_conf * prev_count + verb_conf) / (prev_count + 1),
                        prev_seq_prob,
                        prev_count + 1,
                        prev_rationale,
                    ]
                    break
            if not merged_answer_class:
                # a new class, add the answer to the answer set
                classes.update({ans: [verb_conf, [seq_prob], 1, conf_rationale]})

    # (unique_ans, mean_verb_confidence, mean_seq_prob, count, confidence_rationale)
    stances = []
    #rint("this stance")
    for ans_class, (verb_conf, seq_probs, count, rationale) in classes.items():
        seq_probs_filtered = [seq_prob for seq_prob in seq_probs if seq_prob]
        seq_probs = np.mean(seq_probs_filtered) if seq_probs_filtered else 0
        logging.debug(f"seq_probs_filtered: {seq_probs_filtered}, seq_probs: {seq_probs}")
        stances.append([ans_class, float(verb_conf), float(seq_probs), int(count), rationale])
       #print("yesj")
    logging.debug(f"final ans_set: {stances}")
    return stances


def stance_generation(
    record: pd.Series, agents_mapping: Dict[str, Any], nli_tokenizer: Any, nli_classifier: Any,
) -> Tuple[List[Any], List[Any]]:
    """Stage 1: the selected expert agents vote independently, output a set of semantically unique answers and corresponding confidence/count"""
    votes = []
    for specialization, grouped_agents in agents_mapping.items():
        for agent in grouped_agents:
            #if agent.model_type.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values()) and specialization != "self-ask":
            image = record.get("image", None)
            prob, res = agent.self_deliberate_with_pretrained_instance(record["question"], image=image)
            #rint("hello")
            print(res)
            #else:
            #    prob = None
            #    res = agent.self_deliberate(record["question"])
            vote = tuple((agent.id, agent.model_type, extract_ans(res), extract_conf(res), prob, extract_conf_metrics(res)))
            votes.append(vote)
    logging.info(f"votes: {votes}")
    stances = construct_stances(votes, record["question"], nli_classifier, nli_tokenizer)
    #rint(stances)
    return votes, stances



def revote(
    question: str,
    mappings: List[Tuple[BaseAgent, float, str, str, Image.Image]]
) -> List[Tuple[str, str, str, float, Union[float, None], str]]:
    votings_all = []
    for agent, initial_conf, original_observations, new_observations, image in mappings:
        # 1) Build the prompt exactly as before
        prompt = (
            f"Given the question: '{question}', \n"
            f"{original_observations}\n"
            f"Here are some new observations:\n{new_observations}"
        )
        prompt += "Give your final answer (as short as possible). "
        prompt += (
            "Considering your original belief, group consensus and new observations, "
            "and weighing arguments from multiple sides (including your own), "
            "give rationales for whether you would adjust your original confidence score.\n"
            "Follow this format:\nAnswer:\nRationales:"
        )

        # 2) Send it (with the image) to your VLM
        seq_prob, revote_intermediate = agent.self_deliberate_with_pretrained_instance(
            prompt,
            image=image
        )

        # 3) Extract the rationale
        logging.debug(f"revote_intermediate: {revote_intermediate}")
        rationale = revote_intermediate.split("Rationales:")[-1].strip()

        # 4) Build and send the confidence‐reprompt, again with the image
        conf_prompt = (
            f"Recall your original confidence was {initial_conf:.2f}.\n"
            f"Rationale:\n'''{rationale}'''\n"
            "Provide ONLY the final confidence between 0 and 1.\n"
            "Format exactly:\nConfidence: 0.xxx\n"
            "Confidence: "
            )

        _, revote_conf = agent.self_deliberate_with_pretrained_instance(
            conf_prompt,
            image=image
        )

        # 5) Parse the new confidence
        try:
            conf_parsed = extract_confidence(revote_conf, default=initial_conf)
        except Exception as e:
            logging.debug(f"Confidence parse failed: {e}; raw text: {revote_conf!r}")
            conf_parsed = initial_conf

        # 6) Append exactly the same tuple shape you had before
        votings_all.append((
            agent.id,
            agent.model_type,
            extract_ans(revote_intermediate),
            conf_parsed,
            seq_prob,
            rationale
        ))

    logging.debug(f"votings_all {votings_all}")
    return votings_all
    
def deliberate_with_feedback(
    question: str,
    image: Image.Image,    
    agents: List[BaseAgent],
    nli_tokenizer: Any,
    nli_classifier: Any,
    stance_list: List[Tuple[str, float, float, int, Any]],
    group_pruning: Optional[bool] = False,
    long_form: bool = True,
    self_popularity: bool = False,
    verify: bool = False,
) -> Tuple[Dict[str, str], List[Any], Set[Tuple[str, float, float, int, str]]]:
    """Stage 2: m general agents, each with an assigned stance and corresponding confidence (verb/logit-based). 
    Group deliberation process:

    agents: the list of general agents
    stance_list: [<unique_answer, verb_confidence, model_prob, count, conf_rationale>], sorted by count ascendingly
    """
    agents_observations_mapping = []  # [<agent, initial_conf, original observations, new observations/feedback>]
    class_count = [stance_stats[3] for stance_stats in stance_list]
    # generate one argument for each class
    arguments = {t[0]: None for t in stance_list}  # stance: argument
    if len(stance_list) == 1:
        # if reaching consensus in stage 1, assign only 3 general agents, and no feedback needed
        unique_answer, initial_verb_conf, initial_seq_prob, *_ = stance_list[0]
        initial_conf = max(initial_verb_conf, initial_seq_prob)
        original_observations = (
            f"Your original answer is '{unique_answer}', with a confidence of {initial_conf:.2f}"
        )
        new_observations = "Through deliberation, all other people have agreed with your answer, reaching a consensus."
        agents_observations_mapping = [
            tuple((agents[i], initial_conf, original_observations, new_observations, image)) for i in range(3)
        ]
    else:
        # m general agents, generate arguments and get moderator feedback
        if group_pruning:
            # m = len(agents) < count of independent votes from stage 1 (number of specialized agents that didn't abstain)
            class_freq = class_count / np.sum(class_count)
            logging.debug(f"class_freq: {class_freq}")
            assignment_quantities = np.floor(class_freq * len(agents))
            assignment_quantities = [np.max([quantity, 1]) for quantity in assignment_quantities]
            assignment_quantities[-1] = len(agents) - np.sum(assignment_quantities[:-1])
            assignment_quantities = np.cumsum(assignment_quantities)
        else:
            # exact same assignment as the independent votes from stage 1
            assignment_quantities = np.cumsum(class_count)
        logging.debug(f"assignment_quantities {assignment_quantities}")
        m = assignment_quantities[-1]
        logging.debug(f"agent count: {agents}, stance_list {stance_list} m {m}")

        curr_stance_index = 0
        for index, agent in enumerate(agents):
            # stance_list already sorted
            if index == assignment_quantities[curr_stance_index]:
                curr_stance_index += 1
            if curr_stance_index == len(assignment_quantities):
                break
            assigned_ans, initial_verb_conf, initial_seq_prob, count, _ = stance_list[curr_stance_index]  
            initial_conf = max(initial_verb_conf, initial_seq_prob)         
            agents_observations_mapping.append([agent, assigned_ans, initial_conf, count, image])
            #agents_observations_mapping.append((agent, initial_conf, original_obs, new_obs, image))
            if long_form:
                arguments[assigned_ans] = agent.generate_self_evaluation(question, assigned_ans)
            else:
                if not arguments.get(assigned_ans):
                    arguments[assigned_ans] = agent.generate_argument(question, assigned_ans, image)

        logging.info(f"deliberator arguments: {arguments}")

        ranking = get_collective_feedback(question, arguments, nli_tokenizer, nli_classifier, verify)
        # ranking: [<ans, argument, soundness_score, verbal_feedback>], sorted by soundness_score desc
        logging.debug(f"ranking {question}; {ranking}")

        # construct new observations and update agents_observations_mapping
        for i, mapping in enumerate(agents_observations_mapping):
            agent, assigned_ans, initial_conf, count, image = mapping
            original_observations = (
                f"Your original answer is {assigned_ans}, with a confidence of {initial_conf:.2f}"
            )
            
            for rank, (ans, argument, soundness, feedback) in enumerate(ranking):
                general_feedback = (
                    f"'''{argument}'''\n, which received the following rating and feedback from other deliberators:"
                    f"Soundness score: {soundness:.2f} (ranked {rank+1} out of {len(ranking)})\n"
                    f"Feedback: {feedback}"
                )
                if ans == assigned_ans:
                    feedback_supporting = f"An argument supporting your original answer is\n{general_feedback}"
                    if not self_popularity:
                        feedback_supporting += f"\nNote that {count-1} other {'person' if count == 2 else 'people'} (out of {m}) also agreed with you."
                else:
                    feedback_opposing = f"An argument from the opposing side is\n{general_feedback}"
                    if not self_popularity:
                        feedback_opposing += f"\nNote {m - count} {'person' if m - count == 1 else 'people'} disagreed with you."                   
                    
            self_estimate_popularity = f"Based on the evidence presented, estimate how many deliberators (including yourself, out of {m}) are on your side." if self_popularity else ""
            new_observations = f"Recall that your original confidence was {initial_conf:.2f}\n{feedback_opposing}\n{feedback_supporting}\n{self_estimate_popularity}"
            agents_observations_mapping[i] = tuple((agent, initial_conf, original_observations, new_observations, image))

    # re-voting with new observations (and the corresponding ranking/feedback, if no early consensus)
    final_votes_raw = revote(question, agents_observations_mapping)
    rationales = [vote[-1] for vote in final_votes_raw]
    final_set = construct_stances(
        final_votes_raw, question, nli_classifier, nli_tokenizer, conf_rationales=rationales
    )
    return arguments, final_votes_raw, final_set


def save_vote_history(
    question_id: str,
    question: str, 
    original_votes: List[Any],
    original_stance_list: List[Tuple[str, float, float | None, int, str]],
    final_votes: List[Tuple[str, str, str, float, float | None, str]],
    final_stance_list: List[Tuple[str, float, float | None, int, str]],
    final_majority: Tuple[str, float, int, str],
    output_filepath: Path,
    dataset: str,
):
    vote_keys = ["agent_id", "model", "answer", "verbal_confidence", "sequence_probability", "confidence_metrics"]
    stance_keys = ["answer_class", "avg_verbal_confidence", "avg_sequence_probability", "count", "rationale"]

    original_votes_with_keys = [dict(zip(vote_keys, original_vote)) for original_vote in original_votes]    
    original_stance_list_with_keys = [dict(zip(stance_keys, stance)) for stance in original_stance_list]
    
    vote_keys[-1] = "rationale"
    final_votes_with_keys = [dict(zip(vote_keys, final_vote)) for final_vote in final_votes]
    final_stance_list_with_keys = [dict(zip(stance_keys, stance)) for stance in final_stance_list]

    res = {
        "question": question,    
        "qid": question_id,
        "original_votes": original_votes_with_keys,
        "original_stances": original_stance_list_with_keys,
        "final_votes": final_votes_with_keys,
        "final_stances": final_stance_list_with_keys,
        "final_majority_ans": final_majority[0],
        "final_verbal_confidence": final_majority[1],
    }
    with open(f"{str(output_filepath)}/{dataset}.jsonl", mode="a") as fp:
        fp.write(json.dumps(res, indent=1) + "\n")

def save_agent_info(agent: BaseAgent, dirpath: str = "data/memory/agents/"):
    agent_info = agent.get_agent_parameters()
    api_call = agent.callback_handlers[0]
    api_call_info = dict(
        {
            "Number of successful_requests": api_call.successful_requests,
            "Number of total token": api_call.total_tokens,
            "Number of prompt token": api_call.prompt_tokens,
            "Number of completion token": api_call.completion_tokens,
            "Total cost for this agent": api_call.total_cost,
        }
    )
    output_filepath = f"{dirpath}/info_{agent.id}.json"
    Path(output_filepath).mkdir(parents=True, exist_ok=True)
    with open(f"{str(output_filepath)}/info_{agent.id}.json", "w+") as outfile:
        json.dump(api_call_info | agent_info, outfile)

def agents_deliberation_single_thread(
    df: pd.DataFrame,
    expert_agent_pool: Dict[str, Any],
    general_agent_pool: List[Any],
    nli_tokenizer: Any,
    nli_classifier: Any,
    output_filepath: Path,
    dataset: str,
    long_form: bool = False,
):
    #if not nli_tokenizer or not nli_classifier:
        #nli_tokenizer, nli_classifier = load_entailment_classifier()
    for _, record in tqdm(df.iterrows()):
        # Stage 1
        original_votes, stances = stance_generation(record, expert_agent_pool, nli_tokenizer, nli_classifier)
        if not len(stances):
            logging.info(f"Skip {record['qid']}")
            continue
        original_stance_list = sorted(stances, key=lambda t: t[3])
        # Stage 2
        verify = len(set(["pot", "search"]).intersection(list(expert_agent_pool.keys()))) == 0
        arguments, final_votes, final_ans_set = deliberate_with_feedback(
            record["question"], record["image"],  general_agent_pool, nli_tokenizer, nli_classifier, original_stance_list, long_form=long_form, verify=verify, 
        )
        for i, ans_cls in enumerate(original_stance_list):
            if not ans_cls[-1]:
                original_stance_list[i][-1] = arguments[ans_cls[0]]
        correct_idx    = record["answer"]           # e.g. integer 0/1/2
        final_majority = sorted(list(final_ans_set), key=lambda t: t[2])[-1]
        summary_path = output_filepath / "summary_vqarad_fmm.csv"
        if not summary_path.exists():
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "qid",
                    "predicted_answer",
                    "predicted_confidence",
                    "correct_index",
                ])
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([record["qid"], final_majority[0], final_majority[2], correct_idx])
        print(final_majority[0], final_majority[2])
        logging.debug(f"final majority vote: {final_majority}")
        logging.debug(
            f'Question {record["question"]} → FINAL: "{final_majority[0]}" '
            f'(confidence={final_majority[1]:.2f})'
        )
        logging.debug(f"final majority vote: {final_majority}")
        save_vote_history(
            record["question"],
            record["qid"],
            original_votes,
            original_stance_list,
            final_votes,
            final_ans_set,
            final_majority,
            output_filepath,
            dataset,
        )

        for agent in list(itertools.chain(*expert_agent_pool.values())) + general_agent_pool:
            if agent.model_type in TEST_API_MODELS["openai-chat"]:
                save_agent_info(agent)


def allocate_slots(model_ensemble: bool, group_size: int, validation_data: pd.DataFrame, use_vllm: bool = False):
    # if using model_ensemble (k models), allocation_size = size(expert_agents) // k
    if model_ensemble:
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("Cohere API not found.")
        if not os.getenv("HF_TOKEN"):
            raise ValueError("Huggingface access token for Llama not found.")   
        if not all(
            model.split("/")[-1] in itertools.chain(*TEST_API_MODELS.values()) or model.split("/")[-1] in itertools.chain(*TEST_HF_MODELS.values())
            for model in MODEL_ENSEMBLE_CHOICES
        ):
            raise ValueError("Model not supported.")
        allocation_size = group_size // len(MODEL_ENSEMBLE_CHOICES)
    else:
        allocation_size = group_size

    return allocate_agent_slots(validation_data, allocation_size, use_vllm=use_vllm)


def main(args):
    log_dir = os.path.dirname(args.logfile_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    print(f"Log file created at: {args.logfile_path}")
    # 2) Make sure your memory‐output directory exists _and_ clear out any old JSONL
    os.makedirs(args.memory_filepath, exist_ok=True)
    csv_path = os.path.join(args.memory_filepath, "summary_VQAR_path.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "final_answer", "final_confidence"])
    jsonl_file = os.path.join(args.memory_filepath, f"{args.input_dataset}.jsonl")
    print(f"JSONL file created at: {jsonl_file}")
    logging_level = logging.INFO if args.logging_level == "info" else logging.DEBUG
    logging.basicConfig(
        filename=args.logfile_path,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging_level,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.debug(f"device: {device}")
    # Load environment variables from .env file
    load_dotenv(args.api_key_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    from datasets import load_dataset
    from PIL import Image
    import io
    if args.input_dataset == "scienceqa":
        def open_image_field(img_field):
            if isinstance(img_field, Image.Image):
                return img_field.convert("RGB")
            if isinstance(img_field, dict):
                if img_field.get("path"):
                    return Image.open(img_field["path"]).convert("RGB")
                if img_field.get("bytes"):
                    return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
            return Image.open(img_field).convert("RGB")

        # …

        # 2a) Load the ScienceQA validation split
        ds = load_dataset("derek-thomas/ScienceQA", split="test")
        # 2b) Filter out any examples without images
        ds = ds.filter(lambda ex: ex.get("image") is not None)
        
        # 2c) Turn it into a pandas DataFrame with exactly the fields you need:
        import pandas as pd
        records = []
        for idx, ex in enumerate(ds):
            q = ex["question"].strip()
    

            # build a single "question" string that includes the choices:
            choices_str = "\n".join(
                f"{i+1}. {opt}"
                for i, opt in enumerate(ex["choices"])
            )
            full_prompt = f"{q}\n\nChoices:\n{choices_str}"

            records.append({
                "qid": idx,
                "lecture": ex["lecture"],
                # now question _is_ the full prompt
                "question": full_prompt,
                "answer":ex['answer'],

                # you can still keep the raw list around if you need it later
                "choices": ex["choices"],
                "image": open_image_field(ex["image"]),
                "reference_answers": ex["choices"][ex["answer"]],
            })
        full_df   = pd.DataFrame(records)



        # 1) take your “test” split
        test_data = full_df.iloc[: args.test_sample_size] \
                        .reset_index(drop=True)

        # 2) from that test split, take your “dev” sample for allocation
        dev_data  = test_data.iloc[: args.dev_sample_size] \
                            .reset_index(drop=True)
    # Stage 1 agents
    selection = {"cot": 4}
    expert_agent_pool = populate_expert_agents(
        selection,
        model_ensemble=False,      # turn on multi-model ensemble
        mix_temperature=True,
        use_vllm=args.vllm,
    )

    # Stage 2 agents (same size as Stage 1 agents), fixed for all queries
    general_agent_pool = populate_general_agents(args.group_size)    
    

    logging.debug(f"expert agents: {expert_agent_pool}\ngeneral agents: {general_agent_pool}")

    nli_tokenizer, nli_classifier = load_entailment_classifier()
    nli_classifier.eval()
    df_list = np.array_split(test_data, args.n_thread)
    Path(args.memory_filepath).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.n_thread) as executor:
        futures = [
            executor.submit(
                agents_deliberation_single_thread,
                df,
                expert_agent_pool,
                general_agent_pool,
                nli_tokenizer,
                nli_classifier,                
                Path(args.memory_filepath),
                args.input_dataset,
                args.long_form,
            )
            for df in df_list
        ]
        for future in as_completed(futures):
            logging.info(future.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-logfile_path",
        default="data/logs.log",
        type=str,
        help="logging ouput file",
    )
    parser.add_argument(
        "-logging_level",
        default="info",
        choices=["debug", "info"],
        type=str,
        help="console logging level",
    )
    parser.add_argument(
        "-api_key_path",
        default=".env",
        type=str,
        help="path to the env file with all api keys",
    )
    parser.add_argument(
        "-memory_filepath",
        default="data/memory/",
        type=str,
        help="path to the simulation output",
    )
    parser.add_argument(
        "--agent_ensemble",
        default=True,  # no agent_ensemble for long-form generation tasks
        action="store_true",
        help="whether to ensemble with multiple specialized agents",
    )
    parser.add_argument(
        "--model_ensemble",
        default=False,
        action="store_true",
        help="whether to ensemble with multiple backbone models",
    )
    parser.add_argument(
        "--long_form",
        default=False,  # for long-form generation: do model_ensemble but not agent_ensemble (half ZS half CoT for now)
        action="store_true",
        help="whether the task is long-form generation",
    )
    parser.add_argument(
        "--vllm",
        default=False,
        action="store_true",
        help="whether to use vllm to speed up inference",
    )
    parser.add_argument(
        "-n_thread",
        type=int,
        default=1,
        required=False,
        help="number of threads",
    )
    parser.add_argument(
        "-group_size",
        type=int,
        default=6,
        required=False,
        help="number of expert agents in self-deliberation",
    )
    parser.add_argument(
        "-input_dataset",
        type=str,
        choices=["triviaqa-dev", "sciq-valid", "pathrad", "math-test-prm800k", "gsm8k-test", "WikiLingua-1000-chn-eng", "theoremqa-test", "ambigqa", "gpqa_diamond", "dateUnd", "prfLaw","scienceqa", "Biz-Ethics"],
        help="name of the input dataset",
    )
    parser.add_argument(
        "-dev_sample_size",
        required=False,
        default=2,
        type=int,
        help="size of the sampled development set for agent allocation",
    )
    parser.add_argument(
        "-test_sample_size",
        required=True,
        type=int,
        help="total number of examples to sample for the test set",
    )

    args = parser.parse_args()
    main(args)
