# MSc Individual Project: Towards Unified, Structured and LLM-native 3D Scene Synthesis

Code repo for MSc Individual Project. 

## Quick Start

Preliminaries:
- Install dependencies: `uv sync` and `source .venv/bin/activate`;
- Set up LLM configs in llms.yml and API keys in `.env`;
- Set-up [text-to-3D API](https://github.com/SanBingYouYong/TRELLIS-API) or [retrieval API](https://github.com/SanBingYouYong/shapenet-db) as documented, default to port 8000 and 8001 localhost.

Running the pipeline: 
- Running CSP-based layout search and 3D scene synthesis: `python pipelines/run_csp.py "scene description"`;
- Running direct llm sampling baseline: `python pipelines/run_direct.py "scene description"`;
- Running retrieval-based baseline: change `obtain_method='text-to-3d'` to `'retrieve-shapenet'`.

Checking the results: 
- Inspect the final results in `exp/timestamp/method/final/renders` or opening `final/scene.blend` directly. 

For shape program synthesis in Chapter 4, the code will be released after conference review and will be updated here. 

## Supplementary

Packed archives of evaluation and experiment results are available [here](https://imperiallondon-my.sharepoint.com/:f:/g/personal/sz2224_ic_ac_uk/Ej2Kb61sLvZGif-XJDeGbeUBmka3mee2kfQeC2iiBmcHDA?e=vEwDMP), hosted on Imperial College London OneDrive. Please refer to the [data_readme.md](./data_readme.md) for file references. 
