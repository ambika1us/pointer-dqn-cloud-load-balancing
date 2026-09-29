# Permutation-Equivariant Q-Learning for Variable-Cardinality VM Selection in Cloud Load Balancing

This repository contains the code, data pipeline, and experiment scripts for the paper.

## Overview

We propose a Pointer-Style Deep Q-Network (Pointer-DQN) for cloud load balancing that scores candidate VMs using shared parameters. The architecture is permutation-equivariant and supports variable VM cardinalities without architectural modification.

## Data Sources

The raw traces are not included in this repository due to their size. Please download them from the original sources:

- **Google Borg Cluster Trace 2019**: 
  - Repository: [google/cluster-data](https://github.com/google/cluster-data)
  - Access: The data is available via Google BigQuery. See `ClusterData2019.md` in the repository for access instructions.

- **Alibaba Cluster Trace v2018**: 
  - Repository: [alibaba/clusterdata](https://github.com/alibaba/clusterdata)
  - Access: Fill out the [survey form](http://alibabadeveloper.mikecrm.com/BdJtacN) to receive the download link.

After downloading, place the Alibaba files (`container_meta.csv`, `machine_meta.csv`) into `data/raw/`.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Installation

```bash
git clone https://github.com/<your-username>/rl_load_balancing.git
cd rl_load_balancing
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt