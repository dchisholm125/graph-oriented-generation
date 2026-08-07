# Case Study: Orca Whirlpools Rust Client Config

## Task

Add unit coverage proving that `WhirlpoolDeployment::default()` is the same
deployment as `WhirlpoolDeployment::mainnet()`, including both the program id
and config account.

The correct implementation is a localized test-only change in:

```text
rust-sdk/client/src/config.rs
```

The public API should remain unchanged.

## Repo

- Repository: Orca Whirlpools
- Source: `https://github.com/orca-so/whirlpools`
- Commit: `a119d79bada4e730fef791cac6adb669405a21de`
- Repo state: clean
- Validation: `cargo test --manifest-path rust-sdk/client/Cargo.toml --lib`
- Model: `kimi-k2.6:cloud`
- Run artifact:
  `gog/results/private/paired_orca_rust_config_fair_rag/20260515_165514/orca_rust_client_deployment_default_test`

## Fairness Controls

The benchmark was run as a paired context-provider comparison.

- Same pinned repo commit
- Same task prompt
- Same model
- Same output schema
- Same patch applier
- Same validation command
- Same attempts and trials
- Separate clean worktree per mode
- Only the context provider changed

RAG was tested with stronger fair-fight baselines:

- `rag_vector`
- `rag_keyword`
- `rag_hybrid`

Each RAG mode used a candidate pool of 100 files/chunks and token-budget-aware
packing across budgets:

```text
1000, 4000, 16000, 64000
```

## Result

This is not a "RAG cannot solve it" case. It is a context-quality case.

GOG Membrane passed. Keyword RAG and Hybrid RAG also passed when their retrieval
strategy surfaced the target file. Vector-only RAG failed across all budgets.

GOG's advantage was that it reached full recall with cleaner package-scoped
context, fewer files, lower noise, and fewer total tokens than full-recall
Hybrid RAG.

| Mode | Pass | Total Tokens | Context Precision | Context Recall | Noise Ratio | Retrieved Files | Edited Files | Changed Lines | Failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gog_membrane` | yes | 4,965 | 0.6667 | 1.0 | 0.3333 | 6 | 1 | 17 | none |
| `rag_vector_1000` | no | 1,701 | 0.125 | 0.25 | 0.875 | 8 | 1 | 110 | validation_failure |
| `rag_vector_4000` | no | 5,113 | 0.1176 | 0.5 | 0.8824 | 17 | 1 | 10 | missing_semantic_behavior |
| `rag_vector_16000` | no | 12,609 | 0.0857 | 0.75 | 0.9143 | 35 | 1 | 110 | validation_failure |
| `rag_vector_64000` | no | 16,921 | 0.0732 | 0.75 | 0.9268 | 41 | 1 | 117 | validation_failure |
| `rag_keyword_1000` | yes | 2,943 | 1.0 | 0.25 | 0.0 | 1 | 1 | 13 | none |
| `rag_keyword_4000` | yes | 5,776 | 0.5 | 0.75 | 0.5 | 6 | 1 | 17 | none |
| `rag_keyword_16000` | yes | 11,614 | 0.1875 | 0.75 | 0.8125 | 16 | 1 | 13 | none |
| `rag_keyword_64000` | yes | 19,133 | 0.1034 | 0.75 | 0.8966 | 29 | 1 | 13 | none |
| `rag_hybrid_1000` | yes | 2,920 | 1.0 | 0.25 | 0.0 | 1 | 1 | 13 | none |
| `rag_hybrid_4000` | yes | 5,946 | 0.2857 | 1.0 | 0.7143 | 14 | 1 | 13 | none |
| `rag_hybrid_16000` | yes | 13,429 | 0.1481 | 1.0 | 0.8519 | 27 | 1 | 13 | none |
| `rag_hybrid_64000` | yes | 22,383 | 0.093 | 1.0 | 0.907 | 43 | 1 | 13 | none |

## Best Direct Comparison

The clearest fair comparison is `gog_membrane` vs `rag_hybrid_4000`.

Both passed. Both achieved full context recall. Both edited only
`rust-sdk/client/src/config.rs`.

| Metric | GOG Membrane | Hybrid RAG 4000 |
|---|---:|---:|
| Pass | yes | yes |
| Total tokens spent | 4,965 | 5,946 |
| Context precision | 0.6667 | 0.2857 |
| Context recall | 1.0 | 1.0 |
| Noise ratio | 0.3333 | 0.7143 |
| Retrieved files | 6 | 14 |
| Edited files | 1 | 1 |
| Changed lines | 17 | 13 |

GOG used about 16.5% fewer total tokens than Hybrid RAG 4000:

```text
(5946 - 4965) / 5946 = 16.5%
```

GOG also retrieved less than half as many files and had materially lower noise.

## Retrieved Context

### GOG Membrane

```text
rust-sdk/client/Cargo.toml
rust-sdk/client/package.json
rust-sdk/client/src/config.rs
rust-sdk/client/src/core_types/tick_array.rs
rust-sdk/client/src/gpa/utils.rs
rust-sdk/client/src/lib.rs
```

GOG selected the Rust client package boundary and included the target config
module plus nearby package metadata.

### Hybrid RAG 4000

```text
rust-sdk/client/src/config.rs
rust-sdk/whirlpool/package.json
rust-sdk/client/Cargo.toml
docs/rust/package.json
rust-sdk/macros/package.json
rust-sdk/client/src/pda/whirlpools_config_extension.rs
docs/rust/Cargo.toml
rust-sdk/client/package.json
examples/rust-sdk/whirlpool_repositioning_bot/package.json
docs/rust/src/lib.rs
rust-sdk/client/src/pda/lock_config.rs
rust-sdk/client/src/lib.rs
rust-sdk/core/package.json
Cargo.toml
```

Hybrid RAG found the target file and all gold context, but did so by carrying
substantially more unrelated manifests and neighboring Rust files.

## Patch Quality

### GOG Membrane

GOG added an inline `#[cfg(test)]` module to `rust-sdk/client/src/config.rs`.
It used the existing public methods:

- `WhirlpoolDeployment::default()`
- `WhirlpoolDeployment::mainnet()`
- `.id()`
- `.config_address()`

The patch preserved all existing constructors and API behavior.

Validation result:

```text
24 passed; 0 failed
```

### Passing RAG Modes

The passing keyword and hybrid RAG modes also produced localized test-only
patches once they retrieved `rust-sdk/client/src/config.rs`.

This matters: a strong lexical or hybrid RAG baseline can solve this task. The
GOG claim is not that RAG is incapable. The claim is that GOG served cleaner
context with less noise and lower token spend at full recall.

### Vector RAG Failure

Vector-only RAG failed at every budget.

Observed failure patterns:

- Low-budget vector RAG missed the target source file and rewrote
  `WhirlpoolDeployment`, deleting existing API surface.
- Mid-budget vector RAG edited
  `rust-sdk/client/src/pda/whirlpools_config_extension.rs` instead of
  `rust-sdk/client/src/config.rs`.
- Higher-budget vector RAG retrieved more files but more noise, then rewrote
  deployment semantics and broke downstream PDA tests.

The vector results show that simply increasing context budget can increase
noise without improving semantic patch quality.

## Interpretation

This case study supports a narrower and more credible claim than the original
early run:

```text
GOG does not merely make the prompt smaller.
It can preserve full task recall while reducing package noise.
```

For this Rust SDK task:

- GOG found the intended Rust client package context.
- GOG reached full recall with 6 files.
- Hybrid RAG reached full recall with 14 files at the comparable 4000-token
  budget.
- Keyword RAG solved the task when the file was lexically obvious.
- Vector RAG failed even as the budget increased.

The strongest reading is:

```text
GOG beat vector RAG decisively.
GOG beat fair hybrid RAG on context quality and token efficiency.
GOG did not prove that all RAG baselines fail.
```

## Evidence Artifacts

Primary comparison:

- `gog/results/private/paired_orca_rust_config_fair_rag/20260515_165514/orca_rust_client_deployment_default_test/comparison.md`
- `gog/results/private/paired_orca_rust_config_fair_rag/20260515_165514/orca_rust_client_deployment_default_test/comparison.json`

GOG artifacts:

- `gog_membrane/context.json`
- `gog_membrane/prompt.txt`
- `gog_membrane/model_output.json`
- `gog_membrane/patch.diff`
- `gog_membrane/validation.log`
- `gog_membrane/metrics.json`

Hybrid RAG 4000 artifacts:

- `rag_hybrid_4000/context.json`
- `rag_hybrid_4000/prompt.txt`
- `rag_hybrid_4000/model_output.json`
- `rag_hybrid_4000/patch.diff`
- `rag_hybrid_4000/validation.log`
- `rag_hybrid_4000/metrics.json`

Vector RAG failure artifacts:

- `rag_vector_1000/patch.diff`
- `rag_vector_1000/validation.log`
- `rag_vector_4000/patch.diff`
- `rag_vector_4000/validation.log`
- `rag_vector_16000/patch.diff`
- `rag_vector_16000/validation.log`
- `rag_vector_64000/patch.diff`
- `rag_vector_64000/validation.log`

## Limitations

This is one task on one repository. It is useful as a first private case study,
not as a broad claim about all codebases or all RAG systems.

The task has a lexically obvious target once `config.rs` is discoverable, which
is why keyword RAG performs well. Harder future tasks should test cases where
the target file is less directly named and the necessary context is distributed
across package boundaries.

Future case studies should include:

- multiple trials
- more tasks per repo
- holdout repos
- larger semantic edits
- tasks where tests require cross-file behavior, not only local unit coverage
