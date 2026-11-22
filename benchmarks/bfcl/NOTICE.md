# BFCL data (vendored, unmodified)

The two files under `data/` are copied byte-for-byte from the Berkeley Function
Calling Leaderboard (BFCL) in the Gorilla repository, licensed under the
Apache License 2.0:

- Repository: https://github.com/ShishirPatil/gorilla
- Commit: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (2026-03-23)
- `berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json`
- `berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json`

`pin.json` records the SHA-256 of each file; the importer refuses to run if a
file does not match. To re-fetch and verify the files yourself:

```bash
make bfcl-fetch
```

Only the `simple_python` category is supported. The scorer in
`app/benchmarks/bfcl.py` is a port of the Python path of
`bfcl_eval/eval_checker/ast_eval/ast_checker.py::simple_function_checker` at the
same commit. It is not the official `bfcl_eval` package, so results from this platform are
labeled as not leaderboard-comparable.
