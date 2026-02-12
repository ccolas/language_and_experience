import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
mpi_vars = [var for var in os.environ if 'MPI' in var or 'SLURM' in var or 'PMI' in var]
for var in mpi_vars:
    del os.environ[var]

import re
import numpy as np
import json
import pickle
import sys
repo_path = '/'.join(os.path.abspath(__file__).split('/')[:-4]) + '/'
sys.path.append(repo_path)

from src.game.rules import Rules

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

prompt_path = repo_path + 'data_input/vgdl_doc/'



class StopOnString(StoppingCriteria):
    def __init__(self, stop_string, tokenizer, prompt_len):
        self.stop_token_ids = tokenizer.encode(stop_string)[1:]
        self.length = len(self.stop_token_ids)
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        for slice_end in range(len(input_ids[0]) - self.length + 1):
            if slice_end > self.prompt_len:
                if input_ids[0][slice_end:slice_end + self.length].tolist() == self.stop_token_ids:
                    return True
        return False


# Usage:

class VLLM:
    def __init__(self, model_path):
        # Use full path instead of model_id
        print(f'using model path: {model_path}')

        # Initialize tokenizer with full path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        max_model_len = 2000
        n_gpus = 1 # if '70b' in model_path.lower() else 1

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
            gpu_memory_utilization=0.65,
            max_num_batched_tokens=max_model_len+100,  # Just slightly over max_model_len
            max_num_seqs=1,
            enforce_eager=True,
            max_seq_len_to_capture=max_model_len,  # Explicitly set
        )

        # Set sampling parameters to be memory efficient
        self.sampling_params = SamplingParams(
            max_tokens=500,
            temperature=0.,
            top_k=50,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            ignore_eos=False,
            skip_special_tokens=True,
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

        # self.model = LLM(model=model_path, tensor_parallel_size=n_gpus, max_model_len=max_model_len, enforce_eager=False,
        #                  gpu_memory_utilization=0.65)


    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_loglike(self, prompt, theory, msg):
        assert False
        # Save original generation params
        orig_params = self.generation_params.copy()

        # Set up for single token generation
        self.generation_params.update({
            "max_new_tokens": 1,
            "return_dict_in_generate": True,
            "output_scores": True
        })

        # Prepare prompt
        pre_prompt = self.format_prompt(prompt, theory)
        full_prompt = pre_prompt + msg + '\n'

        # Get token counts
        pre_tokens = self.tokenizer(pre_prompt, return_tensors="pt").input_ids.size(1)
        msg_tokens = self.tokenizer(msg + '\n', return_tensors="pt").input_ids.size(1)

        # Generate and get logprobs
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, **self.generation_params)

        # Calculate logprobs for the message portion
        logprobs = torch.log_softmax(outputs.logits, dim=-1)
        token_ids = inputs.input_ids[0, pre_tokens:pre_tokens + msg_tokens]
        sequence_logprob = sum([logprobs[0, i - 1, token_id].item() for i, token_id in enumerate(token_ids, start=1)])

        # Restore original params
        self.generation_params = orig_params

        return sequence_logprob


    def generate(self, messages):
        # Save original params
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate text
        self.sampling_params.temperature = 0.5
        outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
        self.sampling_params.temperature = 0.

        # Decode only the new part
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

    def format_generation_prompt(self, sys_prompt, generation_prompt, theory_str):
        clean_theory_str = 'Who you are' + theory_str.split('Who you are')[1]
        # Replacement win condition description
        pattern = r'You win the game by either reaching or killing all (.*?) objects.'
        replacement = r'You win the game when all \1 objects are dead.'
        clean_theory_str = re.sub(pattern, replacement, clean_theory_str)
        full_prompt = generation_prompt + clean_theory_str + '\n\nStart with "Message:".'
        messages = [{'role': 'system', "content": sys_prompt},
                    {'role': 'user', 'content': full_prompt}]
        return messages

    def label_results(self, res_path, output_path, ground_truth=False):
        os.makedirs(output_path, exist_ok=True)

        sys_prompt = load(prompt_path + "prompt_system.txt")
        generation_prompt = load(prompt_path + "prompt_generation_chat.txt")
        games = [game for game in os.listdir(res_path)]
        for game in games:
            print(f'GAME: {game}')
            game_path = res_path + game + '/'
            seeds = os.listdir(game_path)
            for seed in seeds:
                print(f"  > seed: {seed}")
                msg_path = output_path + game + f'_{seed}.txt'
                if os.path.exists(msg_path):
                    print('     ALREADY exists')
                    continue
                seed_path = game_path + seed + '/'
                with open(seed_path + 'params.json', 'r') as f:
                    params = json.load(f)
                colors = params['true_game_info']['colors']
                for k, v in colors.items():
                    colors[k] = v.lower()
                if ground_truth:
                    rules = Rules(params, vgdl_script=params['true_game_info']['vgdl_script'])
                else:
                    try:
                        thinking_output = self.load_thinking_output(seed_path)
                    except:
                        continue
                    rules = thinking_output[-1]['best_particle']
                theory_str = rules.str_llm(colors)
                messages = self.format_generation_prompt(sys_prompt, generation_prompt, theory_str)
                msg = self.generate(messages)
                msg_to_print = msg.replace("\n", "\n         ")
                print(f'  > msg: {msg_to_print}')
                with open(msg_path, 'w') as f:
                    f.write(msg)



    def save_prompts(self, res_path, output_path):
        prompts = dict()

        sys_prompt = load(prompt_path + "prompt_system.txt")
        generation_prompt = load(prompt_path + "prompt_generation_chat.txt")

        games = [game for game in os.listdir(res_path)]
        for game in games:
            if game not in prompts:
                prompts[game] = dict()
            game_path = res_path + game + '/'
            seeds = os.listdir(game_path)
            for seed in seeds:
                seed_path = game_path + seed + '/'
                with open(seed_path + 'params.json', 'r') as f:
                    params = json.load(f)
                try:
                    thinking_output = self.load_thinking_output(seed_path)
                except:
                    continue
                colors = params['true_game_info']['colors']
                for k, v in colors.items():
                    colors[k] = v.lower()
                theory_str = thinking_output[-1]['best_particle'].str_llm(colors)
                messages = self.format_generation_prompt(sys_prompt, generation_prompt, theory_str)
                prompts[game][seed] = messages
        with open(output_path, 'w') as f:
            json.dump(prompts, f)
        return prompts


    def load_thinking_output(self, seed_path):
        with open(seed_path + 'params.json', 'r') as f:
            params = json.load(f)
        n_lives_per_gen = params['exp_params']['n_lives_per_gen']
        thinking_outputs_names = [name for name in os.listdir(seed_path + 'dumps/') if 'thinking_output' in name]
        i_lives = []
        for thinking_outputs_name in thinking_outputs_names:
            gen = get_gen(thinking_outputs_name)
            life = get_life(thinking_outputs_name)
            i_lives.append(gen * n_lives_per_gen + life)

        filepath = seed_path + 'dumps/' + thinking_outputs_names[np.argmax(i_lives)]
        with open(filepath, 'rb') as f:
            thinking_output = pickle.load(f)
        return thinking_output

def get_gen(filename):
    return int(filename.split('generation_')[1].split('_life')[0])

def get_life(filename):
    return int(filename.split('_life_')[1].split('_lvl')[0])

def load(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt




if __name__ == '__main__':
    import argparse
    import os

    # Default paths relative to repository
    default_model = os.path.join(repo_path, "data/models/deepseek-coder-1.3b-instruct")
    default_input = os.path.join(repo_path, "data/inference_data/individual/")
    default_output = os.path.join(repo_path, "data_input/descriptions_generated/")

    parser = argparse.ArgumentParser(description='Generate descriptions from trained models using LLM')
    parser.add_argument('--model-path', type=str, default=default_model,
                        help='Path to the LLM model')
    parser.add_argument('--input-dir', type=str, default=default_input,
                        help='Input directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default=default_output,
                        help='Output directory for generated descriptions')
    parser.add_argument('--oracle', action='store_true',
                        help='Generate descriptions from ground truth instead of learned theories')

    args = parser.parse_args()

    if args.oracle:
        print('Generating descriptions from oracle (ground truth) theories')
    else:
        print('Generating descriptions from learned theories')

    llm = VLLM(args.model_path)
    llm.label_results(args.input_dir, output_path=args.output_dir, ground_truth=args.oracle)
