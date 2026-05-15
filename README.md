# Robustness of Graph Construction in Context-Aware IoT Anomaly Detection

This repository contains our project on the paper *Context-Aware Anomaly Detection by Community Detection in the Internet of Things*.

The goal is to study how graph construction choices influence the performance and robustness of a context-aware anomaly detection framework for IoT networks.

## Project Idea

The original paper proposes a framework based on:

- dynamic multi-edge graph construction;
- community detection;
- Heterogeneous Graph Neural Networks;
- edge-level anomaly detection.

Our project focuses on a critical component of the pipeline: **graph construction**.

Rather than modifying the HeteroGNN architecture, we investigate whether the final performance depends strongly on how IoT traffic is represented as a graph.

## Research Question

How much does each edge type contribute to anomaly detection performance?

## Proposed Analysis

We plan to perform an ablation study on the edge types used in the paper:

| Configuration | Edge Types |
|---|---|
| A | Network only |
| B | Network + Context |
| C | Network + Knowledge |
| D | Network + Context + Knowledge |

The objective is to evaluate how different graph representations affect:

- Accuracy;
- Precision;
- Recall;
- F1-score.

## Motivation

The paper reports very high performance, but the framework strongly depends on how network traffic is transformed into a multi-relational graph.  
This makes graph construction a key methodological step for robustness and generalization across different IoT environments.

## Repository Structure

```text
data/        datasets and generated graph files
notebooks/   exploratory and experimental notebooks
src/         reusable Python scripts
results/     tables and figures
slides/      presentation material
docs/        notes and paper analysis
