# Data Setup Guide

This guide explains how to set up the data and models required to run the experiments.

## LLM Models

The social learning and chain conditions require a large language model for processing natural language descriptions. We use Llama 3.1 70B Instruct.

### Downloading the Model

1. Request access to Llama 3.1 from Meta: https://ai.meta.com/llama/

2. Once approved, download the model from HuggingFace:

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Login to HuggingFace
huggingface-cli login

# Download the model
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir /path/to/models/Meta-Llama-3.1-70B-Instruct
```

3. Set the environment variable:

```bash
export MODELS_PATH=/path/to/models
```

Add this to your `~/.bashrc` or `~/.zshrc` for persistence.

### Alternative Models

The code also supports smaller models for testing:

- `deepseek-coder-1.3b-instruct` - Small model for debugging
- `Meta-Llama-3.1-8B-Instruct` - Smaller Llama variant

Specify the model with:
```bash
python run_experiment.py --game beesAndBirds --condition social --llm-model deepseek-coder-1.3b-instruct
```

### GPU Requirements

- 70B model: Requires ~140GB GPU memory (use multiple GPUs or quantization)
- 8B model: Requires ~16GB GPU memory
- 1.3B model: Requires ~4GB GPU memory

## Description Data

Language descriptions are stored in `data_input/descriptions/`. The directory structure is:

```
data_input/descriptions/
├── human_no_feedback/      # Human descriptions from individual learning
│   ├── avoidGeorge_0.txt
│   ├── avoidGeorge_1.txt
│   └── ...
└── machine_no_feedback/    # Model-generated descriptions
    └── ...

```

### Using Your Own Descriptions

To test with custom descriptions:

1. Create a text file with your description
2. Save it as `data_input/descriptions/[source_name]/[game]_[trial].txt`
3. Run with `--msg-source [source_name]`

Example:
```bash
# Create a custom description
echo "The blue objects are enemies. Avoid them to survive." > data_input/descriptions/custom/avoidGeorge_0.txt

# Run with your description
python run_experiment.py --game avoidGeorge --condition social --msg-source custom --trial 0
```

## Example Descriptions

Here are example descriptions for testing (these are synthetic, not from study participants):

### beesAndBirds
```
The game has colorful objects. The yellow things seem safe to collect.
The flying things might be dangerous. Try to collect the good items while avoiding the bad ones.
```

### preconditions
```
You need to collect medicine (green) before you can safely touch the poison (red).
Each medicine gives you protection against one poison.
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors with the 70B model:

1. Use a smaller model for testing
2. Reduce `gpu_memory_utilization` in `src/agent/communicating/llm.py`
3. Use model quantization (e.g., 4-bit quantization with bitsandbytes)

### MPI Errors

If MPI fails to initialize:

1. Ensure MPI is installed: `conda install -c conda-forge mpi4py openmpi`
2. Test MPI: `mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"`
3. For single-CPU testing, set `--n-cpus 1`

### Missing Dependencies

If imports fail:

```bash
pip install -r requirements.txt
pip install -e .  # If using editable install
```
