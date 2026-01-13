# CLI Usage

Vajra includes a rich interactive command-line interface for exploring search.

## Installation

```bash
pip install vajra-bm25[cli]
```

## Basic Commands

### Interactive Mode

```bash
# Start with default BEIR SciFact dataset
vajra-search

# Use a different dataset
vajra-search --dataset beir-nfcorpus
```

### Single Query Mode

```bash
vajra-search -q "machine learning algorithms"
```

### Custom Corpus

```bash
# JSONL file
vajra-search --corpus documents.jsonl

# Single PDF
vajra-search --corpus paper.pdf -q "neural networks"

# Directory of PDFs
vajra-search --corpus ./papers/ -q "attention mechanism"
```

## Command-Line Options

```
vajra-search [OPTIONS]

Options:
  -q, --query TEXT       Single query (non-interactive mode)
  -d, --dataset CHOICE   BEIR dataset: beir-scifact, beir-nfcorpus
  -c, --corpus PATH      Path to corpus (JSONL, PDF, or directory)
  -f, --format FORMAT    Corpus format: jsonl, pdf, pdf_dir
  -k, --top-k INT        Number of results (default: 10)
  --stats                Show index statistics and exit
  --no-rich              Disable rich formatting
  --version              Show version
  --help                 Show help
```

## Interactive Commands

When in interactive mode:

| Command | Description |
|---------|-------------|
| `:help` | Show available commands |
| `:stats` | Display index statistics |
| `:quit` or `:q` | Exit the CLI |
| Any other text | Search query |

## Examples

### Search PDFs Interactively

```bash
$ vajra-search --corpus ./research_papers/

Loading PDF directory corpus from ./research_papers/...
Loaded 20 documents
Building index for 20 documents...
Index built in 492.50 ms

vajra> attention mechanism transformer

Results for: attention mechanism transformer
╭───────┬────────────┬─────────────────────────────────────╮
│ Rank  │      Score │ Document                            │
├───────┼────────────┼─────────────────────────────────────┤
│   1   │     9.2923 │ attention_transformer.pdf           │
│   2   │     8.0073 │ bert.pdf                            │
│   3   │     4.6267 │ gpt2.pdf                            │
╰───────┴────────────┴─────────────────────────────────────╯
Found 3 results in 0.25 ms

vajra> :stats

Index Statistics:
  Documents: 20
  Terms: 12,223
  Non-zero entries: 33,325
  Sparsity: 86.37%

vajra> :quit
```

### Batch Processing with Shell

```bash
# Search multiple queries
for query in "neural networks" "deep learning" "transformers"; do
    vajra-search --corpus data.jsonl -q "$query" --no-rich
done
```

## Output Formats

### Rich Output (default)

Colorful tables with highlighted snippets (requires `rich` package).

### Plain Output

```bash
vajra-search -q "query" --no-rich
```

Simple text output suitable for piping and scripting.
