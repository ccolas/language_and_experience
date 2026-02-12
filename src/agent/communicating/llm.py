"""
LLM Communication Engine for language-guided inference.

This module handles:
- Processing natural language descriptions with LLM
- Computing language likelihoods for theory evaluation
- Generating descriptions from learned theories
"""

import time
import numpy as np
from src.utils import get_repo_path, AVATAR_NAME
import os
import re

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt

class Result:
    def __init__(self, response):
        self.prompt_logprobs = response.choices[0].logprobs['token_logprobs']
        self.text = response.choices[0].text



class CommunicationEngine:
    def __init__(self, time_tracker, params):
        self.params = params
        self.time_tracker = time_tracker
        colors = params['true_game_info']['colors']
        self.colors = dict(zip(colors.keys(), [v.lower() for v in colors.values()]))

        # setup proposal
        prompt_path = get_repo_path() + 'data_input/vgdl_doc/'
        self.sys_prompt = load(prompt_path + 'prompt_system.txt')
        self.prompt_proposal = load(prompt_path + 'prompt_proposal.txt')
        self.prompt_summary = load(prompt_path + 'prompt_summary.txt')
        self.prompt_likelihood = load(prompt_path + 'prompt_likelihood_chat.txt')
        self.prompt_generation = load(prompt_path + 'prompt_generation_chat.txt')
        self.cache_proposal = dict()
        self.cache_summary = dict()
        self.cache_likelihood = dict()
        self.setup_llm(self.params)
        self.total_tokens = 0
        self.best_theory = None  # tracks current best theory
        self.lvl = None

    def reset_cache_proposal(self):
        self.cache_proposal = dict()

    def setup_llm(self, params):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        model_path = params['agent']['thinker']['llm_params']['llm_model']
        print(f'Using LLM model: {model_path}')

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # max_model_len = 4000
        n_gpus = 1
        if '70b' in model_path.lower():
            max_model_len = 4200
            gpu_use = 0.6
        else:
            max_model_len = 4000
            gpu_use = 0.6

        import os
        import torch
        import gc

        # Print initial state
        print("\n=== Initial State ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        memory = 0
        allocated = 0
        max_allocated = 0
        device_count = torch.cuda.device_count()

        for i in range(device_count):
            memory += torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            allocated += torch.cuda.memory_allocated(i) / 1024 ** 3
            max_allocated += torch.cuda.max_memory_allocated(i) / 1024 ** 3
        print(f"Total GPU memory: {memory:.2f} GB")
        print(f"Currently allocated: {allocated:.2f} GB")
        print(f"Max allocated: {max_allocated:.2f} GB")

        # Set allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Clear memory
        print("\n=== Clearing Memory ===")
        torch.cuda.empty_cache()
        gc.collect()
        allocated = 0
        for i in range(device_count):
            allocated += torch.cuda.memory_allocated(i) / 1024 ** 3
        print(f"After clear - Currently allocated: {allocated:.2f} GB")

        # Try loading model
        print("\n=== Loading Model ===")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=n_gpus,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_use,
            max_num_batched_tokens=max_model_len+100,
            max_num_seqs=1,
            enforce_eager=True,
            max_seq_len_to_capture=max_model_len,
            enable_prefix_caching=False,  # Disabled to avoid logprob computation issues
        )

        # Set sampling parameters to be memory efficient
        self.sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.,
            top_k=50,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            ignore_eos=False,
            skip_special_tokens=True,
            prompt_logprobs=1
            # stop_token_ids=stop_token_ids
        )

        # Print final state
        print("\n=== Final State ===")
        allocated = 0
        max_allocated = 0
        for i in range(device_count):
            allocated += torch.cuda.memory_allocated(i) / 1024 ** 3
            max_allocated += torch.cuda.max_memory_allocated(i) / 1024 ** 3
        print(f"Final allocated: {allocated:.2f} GB")
        print(f"Final max allocated: {max_allocated:.2f} GB")


    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_logprob(self, user_prompts, start_assistant_prompt, eval_prompts, keep_eot_token):
        if not isinstance(user_prompts, list):
            user_prompts = [user_prompts]
        if not isinstance(eval_prompts, list):
            eval_prompts = [eval_prompts]

        n_tok_inputs = []
        all_messages = []
        for use_prompt, eval_prompt in zip(user_prompts, eval_prompts):
            messages_pre = [{'role': 'system', "content": self.sys_prompt},
                            {'role': 'user', 'content': use_prompt},
                            {'role': 'assistant', 'content': start_assistant_prompt}]
            messages_template_pre = self.tokenizer.apply_chat_template(messages_pre, tokenize=False, add_generation_prompt=False)
            n_tok_inputs.append(self.count_tokens(messages_template_pre) - 1)  # here we have an EOT token to remove
            messages = [{'role': 'system', "content": self.sys_prompt},
                        {'role': 'user', 'content': use_prompt},
                        {'role': 'assistant', 'content': start_assistant_prompt + eval_prompt}]
            messages_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            all_messages.append(messages_template)
        # print(all_messages[-1])
        results = self.model.generate(all_messages, self.sampling_params, use_tqdm=False)
        n_tokens_outputs = [len(res.prompt_token_ids) - n_tok_pre for res, n_tok_pre in zip(results, n_tok_inputs)]
        self.total_tokens += np.sum([n_tok_out + n_tok_in for n_tok_out, n_tok_in in zip(n_tokens_outputs, n_tok_inputs)])
        try:
            if keep_eot_token:
                # print(repr(self.tokenizer.decode(results[0].prompt_token_ids[-n_tokens_outputs[0]:])))
                logprob = [sum([list(info.values())[0].logprob for info in res.prompt_logprobs[-n_tok_out:]]) for res, n_tok_out in zip(results, n_tokens_outputs)]
            else:
                # print(repr(self.tokenizer.decode(results[0].prompt_token_ids[-n_tokens_outputs[0]:-1])))
                logprob = [sum([list(info.values())[0].logprob for info in res.prompt_logprobs[-n_tok_out:-1]]) for res, n_tok_out in zip(results, n_tokens_outputs)]
        except Exception as e:
            print(f"Error computing logprobs: {e}")
            print(f"n_tokens in prompt: {len(self.tokenizer.encode(eval_prompts[0]))}")
            raise RuntimeError("Failed to compute logprobs. Ensure prefix_caching is disabled.")

        return np.array(logprob)


    # methods to get linguistic proposal distribution

    def format_prompt_proposal(self, linguistic_data, obj_names, key, candidates, rewritten_msg_without_objs, rewritten_msg_with_objs):
        # msg_str = self.format_linguistic_data(linguistic_data, prefix='Message from the player:')
        rewritten_msg_without_objs = f'Message from the player:\n"{rewritten_msg_without_objs}"'
        rewritten_msg_with_objs = f'Message from the player:\n"{rewritten_msg_with_objs}"'
        candidate_prompt = "\n".join([f'{idx + 1}) {cand}' for idx, cand in enumerate(candidates)])
        if self.best_theory and not isinstance(key, str):
            objs_known_so_far = self.best_theory.str_obj_llm(self.colors)
        else:
            objs_known_so_far = f"Known Objects: {self.format_obj_names(obj_names)}"
        if 'win' in key:
            key = key[4:]
            user_prompt = (f"{self.prompt_proposal}"
                           f"{objs_known_so_far}\n\n"
                           f"{rewritten_msg_with_objs}\n\n"
                           f"Do you need to kill {self.colors[key]} objects to win?\n"
                           f"{candidate_prompt}")
            stop = 1
        elif 'lose' in key:
            key = key[5:]
            user_prompt = (f"{self.prompt_proposal}"
                           f"{objs_known_so_far}\n\n"
                           f"{rewritten_msg_with_objs}\n\n"
                           f"Do you lose if {self.colors[key]} objects die?\n"
                           f"{candidate_prompt}")
            stop = 1
        elif isinstance(key, str):
            user_prompt = (f"{self.prompt_proposal}"
                           f"{objs_known_so_far}\n\n"
                           f"{rewritten_msg_without_objs}\n\n"
                           f"How do {self.colors[key]} objects behave?\n"
                           f"{candidate_prompt}")
            stop = 1
        elif isinstance(key, tuple):
            user_prompt = (f"{self.prompt_proposal}"
                           f"{objs_known_so_far}\n\n"
                           f"{rewritten_msg_with_objs}\n\n"
                           f"What happens to {self.colors[key[0]]} objects when they collide with {self.colors[key[1]]} objects?\n"
                           f"{candidate_prompt}\n")
            stop = 1
        else:
            raise NotImplementedError
        start_assistant_prompt = (f"Reasoning: .......................\n\n"
                                  f"Answer (pick one from the list):\n")
        return user_prompt, start_assistant_prompt

    def format_prompt_analysis(self, linguistic_data, obj_names, with_objects):
        if self.best_theory and with_objects:
            objs_known_so_far = self.best_theory.str_obj_llm(self.colors)
        else:
            objs_known_so_far = f"Known Objects:\n{self.format_obj_names(obj_names).replace('obj_', '')}"
        objs_known_so_far = objs_known_so_far.replace("Known Objects:", f"Known Objects (at level {self.lvl}):")
        msg_str = self.format_linguistic_data(linguistic_data, prefix='Original Message:')
        user_prompt = (f"{self.prompt_summary}"
                       f"{objs_known_so_far}\n\n"
                       f"{msg_str}\n\n"
                       f"Please analyze and rewrite the message.")
        return user_prompt, msg_str, objs_known_so_far

    def generate_prompt_analysis(self, user_prompt, msg_str):
        user_prompt = user_prompt.replace("Known Objects:", f"Known Objects (at level {self.lvl}):")
        messages = [{'role': 'system', "content": self.sys_prompt},
                    {'role': 'user', 'content': user_prompt}]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Generate text
        self.sampling_params.temperature = 0.5
        self.sampling_params.max_tokens = 500
        outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
        self.sampling_params.max_tokens = 1
        self.sampling_params.temperature = 0.
        generated_text = outputs[0].outputs[0].text
        generated_text = generated_text.replace('\n\n', '\n')

        # extract analysis and rewritten message:
        analysis, message = None, None
        for split_str in ['Rewritten Message:', 'Rewritten message:', 'Message:', 'message:', 'Message', 'message']:
            if split_str in generated_text:
                analysis, message = generated_text.split(split_str)[:2]
                break
        if analysis is None:
            analysis = generated_text
            message = msg_str.split('Original Message:')[1]
        message = message.replace('"', '').strip()
        analysis = analysis.strip()
        return analysis, message

    def get_linguistic_score(self, prompt_to_eval, linguistic_data, obj_names, candidates, key):

        self.time_tracker.tic('get_linguistic_score')
        # rewrite the message without objects
        user_prompt, msg_str, objs_known_so_far = self.format_prompt_analysis(linguistic_data, obj_names, with_objects=False)
        if user_prompt not in self.cache_summary.keys():
            analysis, message = self.generate_prompt_analysis(user_prompt, msg_str)
            self.cache_summary[user_prompt] = (analysis, message)
            print(f"\nNew message parsing (#{len(self.cache_summary)}, prop, without objs):\n\nTrue message:\n{msg_str}\n\n{objs_known_so_far}\n\nAnalysis:\n{analysis}\n\nMessage:\n{message}\n\n")
            stop = 1
        rewritten_msg_without_objs = self.cache_summary[user_prompt][1]
        # rewrite the message with objects
        user_prompt, msg_str, objs_known_so_far = self.format_prompt_analysis(linguistic_data, obj_names, with_objects=True)
        if user_prompt not in self.cache_summary.keys():
            analysis, message = self.generate_prompt_analysis(user_prompt, msg_str)
            self.cache_summary[user_prompt] = (analysis, message)
            print(f"\nNew message parsing (#{len(self.cache_summary)}, prop, with objs):\n\nTrue message:\n{msg_str}\n\n{objs_known_so_far}\n\nAnalysis:\n{analysis}\n\nMessage:\n{message}\n\n")
            stop = 1
        rewritten_msg_with_objs = self.cache_summary[user_prompt][1]

        user_prompt, start_assistant_prompt = self.format_prompt_proposal(linguistic_data, obj_names, key, candidates, rewritten_msg_without_objs, rewritten_msg_with_objs)
        prompt_to_eval = f"{candidates.index(prompt_to_eval) + 1})"
        key_lang = user_prompt + start_assistant_prompt + prompt_to_eval
        if key_lang not in self.cache_proposal.keys():
            self.cache_proposal[key_lang] = self.get_logprob(user_prompt, start_assistant_prompt, prompt_to_eval, keep_eot_token=False)[0]  # add
        lang_logprob = self.cache_proposal[key_lang]
        score = lang_logprob
        self.time_tracker.toc('get_linguistic_score')
        return score

    def compute_loglike_from_theories(self, theories, linguistic_data, obj_names):
        user_prompt, msg_str, objs_known_so_far = self.format_prompt_analysis(linguistic_data, obj_names, with_objects=True)
        if user_prompt not in self.cache_summary.keys():
            analysis, message = self.generate_prompt_analysis(user_prompt, msg_str)
            self.cache_summary[user_prompt] = (analysis, message)
            print(f"\nNew message parsing (#{len(self.cache_summary)}, like):\n\nTrue message:\n{msg_str}\n\n{objs_known_so_far}\n\nAnalysis:\n{analysis}\n\nMessage:\n{message}\n\n")
        eval_prompt = self.cache_summary[user_prompt][1]
        user_prompts, start_assistant_prompt = self.format_prompt_likelihood(theories)
        eval_prompts = [eval_prompt] * len(user_prompts)
        logprobs = self.get_logprob(user_prompts, start_assistant_prompt, eval_prompts, keep_eot_token=True)
        return logprobs

    def generate_description(self, theory):
        messages = self.format_prompt_generation(theory)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Generate text
        self.sampling_params.temperature = 0.5
        self.sampling_params.max_tokens = 500
        outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
        self.sampling_params.max_tokens = 1
        self.sampling_params.temperature = 0.

        # parse text
        generated_text = outputs[0].outputs[0].text
        str_to_clean = ['#', '**']
        for c in str_to_clean:
            generated_text = generated_text.replace(c, '')
        parsed_text = generated_text
        if 'Message:' in parsed_text:
            parsed_text = parsed_text.split('Message:')[1]
        elif 'Message' in parsed_text:
            parsed_text = parsed_text.split('Message')[1]
        return parsed_text.strip()


    def format_prompt_generation(self, theory):
        theory_str = theory.str_llm(self.colors)
        clean_theory_str = 'Who you are' + theory_str.split('Who you are')[1]
        # Replacement win condition description
        pattern = r'You win the game by either reaching or killing all (.*?) objects.'
        replacement = r'You win the game when all \1 objects are dead.'
        clean_theory_str = re.sub(pattern, replacement, clean_theory_str)
        full_prompt = self.prompt_generation + clean_theory_str + '\nStart with "Message:".'
        messages = [{'role': 'system', "content": self.sys_prompt},
                    {'role': 'user', 'content': full_prompt}]
        theory.prompt = theory_str
        return messages

    def format_prompt_likelihood(self, theories):
        user_prompts = []
        for theory in theories:
            theory_str = theory.str_llm(self.colors)
            clean_theory_str = 'Who you are' + theory_str.split('Who you are')[1]
            # Replacement win condition description
            pattern = r'You win the game by either reaching or killing all (.*?) objects.'
            replacement = r'You win the game when all \1 objects are dead.'
            clean_theory_str = re.sub(pattern, replacement, clean_theory_str)
            user_prompt = self.prompt_likelihood + clean_theory_str + '\nStart with:\n\n"""\nReasoning: .......................\n\nMessage:\n[message goes here]\n"""'
            user_prompts.append(user_prompt)
            theory.prompt = theory_str
        start_assistant_prompt = 'Reasoning: .......................\n\nMessage:\n'
        return user_prompts, start_assistant_prompt

    def format_obj_names(self, obj_names):
        s = ''
        for obj_name in sorted(obj_names):
            if s != '':
                s += ', '
            s += f'obj_{self.colors[obj_name]}'
            if AVATAR_NAME in obj_name:
                s += ' (avatar)'
            if obj_name == 'wall':
                s += ' (wall)'
        return s

    def format_linguistic_data(self, linguistic_data, prefix=""):
        assert len(linguistic_data) > 0
        s = prefix
        for msg in linguistic_data:
            if s != '':
                s += "\n"
            s += f"{msg}"
        while s[-1] =='\n':
            s = s[:-1]
        return s
