# gpt-cognitive-dissonance project

replication and extension of [Lehr et al., 2025](https://www.pnas.org/doi/abs/10.1073/pnas.2501823122?af=R)

usage:
- call `experiment.py` to run the experiment

requires:
- `config.json` file, appropriately structured
- `.env` file with OPENAI_API_KEY set

arguments:
- `--interleaved` randomizes the order in which trials are run (default is to run sequentially in `config.json` order)
- `max_workers=k` enables multithreading across `k` threads
