# BMP and Structure-aware PU-GNN source recovery — 2026-08-09

> Status update: BMP is inactive provenance. It was removed from the active
> registry and launchers as `EXCLUDED_NO_RECOVERABLE_OFFICIAL_SOURCE`. The
> mathematical core below is retained only to preserve the historical audit;
> no training bridge, environment, sanity run, or benchmark result is planned.

## BMP

The AAAI 2026 proceedings page and paper identify the paper, authors, DOI
`10.1609/aaai.v40i19.38671`, and advertise
`https://github.com/Thankstaro/BMP` as both code and extended-version location.
At audit time:

- Git clone/ls-remote requested credentials, consistent with an unavailable or
  non-public repository.
- GitHub REST `repos/Thankstaro/BMP` returned HTTP 404.
- The author's public-repository listing contained only `G-2Former`, not BMP.
- GitHub repository searches for the exact title, advertised URL, and title
  fragments returned zero repositories.
- General web, author, fork, institutional, and paper-index searches found no
  recoverable mirror or author-linked checkout.

The inactive provenance record therefore remains `PAPER_DERIVED`, not author official. The local
module implements equations 3–10: probability ordering, upper/lower normalized
routing, independent BMP trees, the BMP forest, masked consistency, supervised
loss, and mask regularization. `T=50` and forest order 3 are recorded from the
paper. It is not an active reproduction candidate.

## Structure-aware PU-GNN

The CIKM 2023 paper is arXiv `2310.13538v1`, DOI
`10.1145/3583780.3615250`. Source checks covered the paper and arXiv pages,
Hansi Yang, Yongqi Zhang, Quanming Yao/LARS pages and repositories, GitHub
repository search by exact title/arXiv ID/distance-aware loss/PU-GNN, and Papers
with Code. The paper contains no code URL; Papers with Code reports no
implementation; exact searches returned no matching repository.

The registry therefore uses `PAPER_DERIVED`, class `PU_AUXILIARY`, and
`fair_ranking: false`. The local module implements paper equation 2
distance-aware PU loss, equation 3 structural regularizer with non-neighbor
sampling, and defaults alpha=0.01, delta=3, K=50, near/far priors 0.6/0.3,
two-layer GCN hidden size 16. It adds no CoReGAD component. Formal reproduction
remains blocked behind native sanity.

## Search-tool errors retained

The paper-search CLI returned arXiv results but reported these dependency
failures verbatim: `openreview not installed`; OpenAlex, DBLP, Crossref, and
Semantic Scholar unavailable because the local CLI environment lacked
`requests`. Direct primary-source and GitHub checks were then used for the
source-recovery decision.
