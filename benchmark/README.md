# Normal-only OOF baseline protocol

The active repository-verified pool is limited to GGAD, RHO, GraphNC, PAGE,
TAQ-GAD, TAM, HUGE, OCGNN, and GAD-NR. Third-party code is not vendored here;
each rebuildable patch records the official URL, frozen upstream SHA, and patch
SHA256 in [`patches/manifest.json`](patches/manifest.json).

`ACTIVE` means the author/official repository identity and current
reachability passed the hard gate. It does not hide protocol blockers. PAGE's
official archive currently states that training code is not released and only
reloads checkpoints. TAQ-GAD's released runner uses ground-truth validation
metrics for checkpoint selection. They are therefore explicitly marked
`PROTOCOL_INCOMPATIBLE`, not silently treated as same-protocol results.

GGAD, RHO, GraphNC, TAM, HUGE, OCGNN, and GAD-NR remain `ADAPTER_REQUIRED`
until their native optimizer is connected to exact node-ID manifests without
in-loop label reads. No published metric is copied into the formal result
table, and no formal baseline training was run during this preparation.

The normative rules are in
[`NORMAL_ONLY_OOF_PROTOCOL.md`](protocol/NORMAL_ONLY_OOF_PROTOCOL.md). AUPRC
is primary; AUROC, Recall@K, Precision@K, and NDCG@K are secondary, with `K`
equal to the anomaly count in the fixed evaluation mask.
