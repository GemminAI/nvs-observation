# nvs-observation

Observation-side package for NVS. It provides the capture-record vocabulary
(`nvs_observation.contracts`), the checks and mapping that turn a capture into an
observation contract, and an optional integration with `nvs_edge`.

- Requires Python 3.12 or newer (3.12, 3.13, 3.14 are supported) and NumPy.
- Install: `pip install nvs-observation`.
- Without `nvs_edge`, everything that does not measure through it still works: the capture
  record, its checks and the mapping. Anything that needs `nvs_edge` reports that it is
  unavailable (a `DEFERRED` result) and does not raise.
- `nvs_observation.contract_check` verifies a contract v2 document and recomputes its hash
  through `nvs_edge`; without `nvs_edge` a document is reported as `NOT_VERIFIED`.
- With `nvs_edge` installed, `EdgeObserver` measures a session through it. The names this
  package reads from `nvs_edge` are listed in `kernel-api-registry.json`; the tests check that
  the list matches the sources.

Licensed under Apache-2.0 (see `LICENSE` and `NOTICE`).
