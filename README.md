# AI Engineering & Learning Record

This repository is my private learning record for AI engineering, algorithms, and machine learning foundations. It keeps course notes, small implementations, and practice logs in one place while mature projects can later move into standalone repositories.

## Current Tracks

| Track | Focus | Expected output |
| --- | --- | --- |
| [Agent Engineering](tracks/agent-engineering/) | LangChain, LangGraph, MCP, RAG, tools, evaluation | Small demos, reading notes, reusable agent patterns |
| [Algorithms & LeetCode](tracks/algorithms-leetcode/) | Java, data structures, coding patterns, interview practice | Problem notes, Java implementations, pattern summaries |
| [Machine Learning](tracks/machine-learning/) | Classical ML implementations and ML theory review | From-scratch models, experiment notes, theory summaries |

## Repository Layout

```text
tracks/
├── agent-engineering/
├── algorithms-leetcode/
└── machine-learning/
    └── classical-ml/
        ├── Bayes/
        ├── KNN/
        └── Version1/
```

## Study Direction

- Use `tracks/agent-engineering/` for the main LLM Agent learning line: LangChain first, then LangGraph, then MCP.
- Use `tracks/algorithms-leetcode/` for Java-based algorithm practice and coding interview preparation.
- Use `tracks/machine-learning/classical-ml/` to keep existing classical machine learning code and gradually add clearer notes.
- When a learning demo becomes useful as a portfolio project, promote it into a separate public repository.

## Environment

```bash
pip install -r requirements.txt
```

Some legacy scripts may need to be run from their own folders because their data paths were written for the original course structure.
