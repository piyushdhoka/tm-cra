# Bibliography verification report

Date of verification: Oct. 7, 2026. Output file: `refs_verified.bib` (55 entries: the 53 requested keys, i.e. items 1-51 plus the two keys under item 52, plus `taori2023alpaca` for the original Stanford Alpaca requested under item 40, and one peer-reviewed alternative `xu2024beyondperplexity` for item 51).

Method: every arXiv ID was resolved through the arXiv API (`export.arxiv.org/api/query`), which returns the canonical title, author list, dates and author comments; venues were then confirmed on the publisher side (ACL Anthology pages/.bib, PMLR volume pages, proceedings.neurips.cc / papers.nips.cc, proceedings.iclr.cc and iclr.cc poster pages, proceedings.mlsys.org, Crossref for DOIs, Zenodo, Hugging Face dataset cards + HF API, GitHub READMEs). OpenReview (site and API) and DBLP were unreachable (bot challenge / 403), so ICLR acceptances were confirmed on the official ICLR sites instead. Semantic Scholar was used only as a cross-check, never as the sole source.

Summary: all 55 entries VERIFIED at a primary source (54 fully; `olmo2025olmo2` is verified as an arXiv preprint, but its COLM 2025 venue rests only on the authors' arXiv comment, see below). No entry needed an `UNVERIFIED` placeholder.

## Per-key status

| # | Key | Status | Primary source(s) | Corrections / notes |
|---|-----|--------|-------------------|---------------------|
| 1 | men2024shortgpt | VERIFIED | https://aclanthology.org/2025.findings-acl.1035/ ; arXiv API 2403.03853 | **Venue corrected**: Findings of **ACL 2025** (not NAACL 2025), pp. 20192-20204, DOI 10.18653/v1/2025.findings-acl.1035. The published version has 9 authors (adds Qianhao Yuan vs. 8 on arXiv); the Anthology list is used. |
| 2 | gromov2024unreasonable | VERIFIED | https://proceedings.iclr.cc/paper_files/paper/2025/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html ; https://iclr.cc/virtual/2025/poster/28400 ; arXiv API (v2 "ICLR camera-ready") | ICLR 2025 confirmed. |
| 3 | kim2024shortened | VERIFIED | arXiv API 2402.02834 (comment: v1 accepted at ICLR 2024 Workshop ME-FoMo, OpenReview id 18VGxuOdpu) ; https://github.com/Nota-NetsPresso/shortened-llm (README citation block) | Cited as the ME-FoMo workshop paper with its workshop title "Shortened LLaMA: A Simple Depth Pruning for Large Language Models". **Note**: arXiv v2 (June 2024) was retitled "Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods"; the OpenReview page itself could not be fetched (bot challenge). |
| 4 | song2024sleb | VERIFIED | https://proceedings.mlr.press/v235/song24f.html | ICML 2024, PMLR 235:46136-46155. |
| 5 | chen2024streamline | VERIFIED | https://proceedings.iclr.cc/paper_files/paper/2025/hash/4b00a351b41358965613c118e87dc28c-Abstract-Conference.html ; https://iclr.cc/virtual/2025/poster/30185 | ICLR 2025 confirmed. |
| 6 | muralidharan2024minitron | VERIFIED | https://proceedings.neurips.cc/paper_files/paper/2024/hash/4822991365c962105b1b95b1107d30e5-Abstract-Conference.html | NeurIPS 2024 (vol. 37). No page numbers on the proceedings site, so none given. |
| 7 | sreenivas2024minitronpractice | VERIFIED | arXiv API 2408.11796 (v4) | arXiv-only. 22 authors; truncated after 12 with "and others" (v4 author order used). |
| 8 | lu2024reassessing | VERIFIED | arXiv API 2411.15558 | arXiv-only (no venue found). |
| 9 | linearpatch2025 | VERIFIED | https://papers.nips.cc/paper_files/paper/2025/hash/e323ebf7c91a4ca061612ba4a2f2164d-Abstract-Conference.html ; arXiv API | **Title corrected**: "A Simple Linear Patch Revives Layer-Pruned Large Language Models" (Chen et al.); **venue added**: NeurIPS 2025 (vol. 38). |
| 10 | clp2025 | VERIFIED | arXiv API 2510.23652 | **Title corrected**: "The Structural Scalpel: Automated Contiguous Layer Pruning for Large Language Models" (Lu et al.); "CLP" is the method name. arXiv-only. |
| 11 | ghosted2026 | VERIFIED | arXiv API 2605.15491 | Full title: "Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs" (Yun et al.). arXiv-only. |
| 12 | limits2026layer | VERIFIED | arXiv API 2602.01997 (v4, Aug. 2026) | Title ends "...in Large Language Models" (not "LLMs"). Shrestha et al. arXiv-only; no venue comment on the abstract page. |
| 13 | smarthome2026 | VERIFIED | arXiv API 2609.17515 | Full title: "What Breaks Under Pruning in Smart Homes, and When? Evaluating LLM Degradation Across Architectures and Task Complexity" (Zhang et al.). Comment says "Submitted to EACL Industry Track" (not accepted), so cited as arXiv preprint. |
| 14 | he2026paser | VERIFIED | https://iclr.cc/virtual/2026/poster/10007567 ; arXiv API (comment "Accepted by ICLR 2026") | ICLR 2026 poster. arXiv spells the 3rd author "Hui-Ling Zhen" (ICLR page: "Huiling Zhen"); arXiv form kept. |
| 15 | thangarasa2024selfdata | VERIFIED | https://proceedings.mlsys.org/paper_files/paper/2025/hash/af2d9fb5bcee19ef2dfa70d843520c97-Abstract-Conference.html ; https://mlsys.org/virtual/2025/3247 | **Venue added**: Proceedings of Machine Learning and Systems 7 (MLSys 2025). (Also appeared at a NeurIPS 2024 workshop.) |
| 16 | shortopd2026 | VERIFIED | arXiv API 2607.13124 | Full title: "ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation" (Zhang et al.). arXiv-only. |
| 17 | xia2024sheared | VERIFIED | https://proceedings.iclr.cc/paper_files/paper/2024 (listing) ; PDF https://proceedings.iclr.cc/paper_files/paper/2024/file/160adf2dc118a920e7858484b92a37d8-Paper-Conference.pdf | ICLR 2024 confirmed. |
| 18 | xie2023doremi | VERIFIED | https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html | NeurIPS 2023 (vol. 36). |
| 19 | ma2023llmpruner | VERIFIED | https://proceedings.neurips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html | NeurIPS 2023 (vol. 36). |
| 20 | rankadaptor2024 | VERIFIED | https://aclanthology.org/2025.findings-naacl.321/ (+ .bib) | Findings of NAACL 2025 confirmed, pp. 5796-5810, DOI 10.18653/v1/2025.findings-naacl.321. |
| 21 | restorelcc2025 | VERIFIED | https://proceedings.neurips.cc/paper_files/paper/2025/hash/099ba28b20462dac7bf057d21c27a011-Abstract-Conference.html ; arXiv API (NeurIPS 2025 Spotlight) | NeurIPS 2025 (vol. 38). The proceedings page shows profile-style names ("Chua Deryl", "Gee Ng"); the full names from arXiv ("Jia Jim Deryl Chua", "Gee Wah Ng") are used. |
| 22 | hu2022lora | VERIFIED | https://iclr.cc/virtual/2022/poster/6319 ; arXiv API | ICLR 2022. |
| 23 | dettmers2023qlora | VERIFIED | https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html | NeurIPS 2023 (vol. 36). |
| 24 | llama32pruning2025 | VERIFIED | arXiv API 2512.22671 | Title/author: "Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2", single author Pere Martra. arXiv-only. |
| 25 | freelunch2025 | VERIFIED | arXiv API 2510.14444 ; https://arxiv.org/abs/2510.14444 | Title: "A Free Lunch in LLM Compression: Revisiting Retraining after Pruning" (Wagner, Roux, Zimmer, Pokutta). A web summary claimed ICML acceptance, but neither the arXiv page nor icml.cc confirms it, so cited as arXiv preprint. |
| 26 | taskpruning2026 | VERIFIED | arXiv API 2604.27115 | Title: "Exploring the Limits of Pruning: Task-Specific Neurons, Model Collapse, and Recovery in Task-Specific Large Language Models" (Siam et al.). arXiv-only. |
| 27 | geng2025jsonschemabench | VERIFIED | arXiv API 2501.10868 (v3) | **Title corrected** to the current arXiv title "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models" (the v1 title was "Generating Structured Outputs from Language Models: Benchmark and Studies"). No published venue found; arXiv preprint. |
| 28 | patil2025bfcl | VERIFIED | https://proceedings.mlr.press/v267/patil25a.html | ICML 2025, PMLR 267:48371-48392; 7 authors as listed by PMLR. No arXiv ID exists for this paper. |
| 29 | zhou2023ifeval | VERIFIED | arXiv API 2311.07911 | arXiv-only. |
| 30 | cobbe2021gsm8k | VERIFIED | arXiv API 2110.14168 | arXiv-only. |
| 31 | rottger2024xstest | VERIFIED | https://aclanthology.org/2024.naacl-long.301/ | NAACL 2024 long, pp. 5377-5400, DOI 10.18653/v1/2024.naacl-long.301. Anthology lists "Hannah Kirk" (arXiv: "Hannah Rose Kirk"); Anthology form used. |
| 32 | hendrycks2021mmlu | VERIFIED | https://iclr.cc/virtual/2021/poster/2962 ; arXiv API ("ICLR 2021") | ICLR 2021. |
| 33 | guo2017calibration | VERIFIED | https://proceedings.mlr.press/v70/guo17a.html | ICML 2017, PMLR 70:1321-1330. |
| 34 | gao2024lmeval | VERIFIED | https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/CITATION.bib ; https://zenodo.org/records/10256836 | The repository's own CITATION.bib (current main) cites v0.4.0, Dec. 2023, DOI 10.5281/zenodo.10256836, title "A framework for few-shot language model evaluation" - so the entry carries **year 2023** despite the key. A later Zenodo record v0.4.3 (Jul. 2024, DOI 10.5281/zenodo.12608602, https://zenodo.org/records/12608602) also exists if you prefer a 2024 date; its Zenodo author list is the GitHub-contributor list, not the Gao et al. list. 23 authors truncated after 12. |
| 35 | grattafiori2024llama3 | VERIFIED | arXiv API 2407.21783 | 561 authors; truncated after 12 with "and others". arXiv-only. |
| 36 | qwen2025qwen25 | VERIFIED | arXiv API 2412.15115 | arXiv lists the author field as "Qwen, :, An Yang, ..."; entry uses "{Qwen} and Yang, An and ... and others". **Year set to 2024** (v1 Dec. 19, 2024; arXiv ID 2412), although the key says 2025. |
| 37 | allal2025smollm2 | VERIFIED | arXiv API 2502.02737 | arXiv-only (no COLM/other venue found). 22 authors truncated after 12. |
| 38 | olmo2025olmo2 | VERIFIED (preprint); venue note partially verified | arXiv API 2501.00656 (v3 comment: "Shorter version accepted to COLM 2025") | Author field on arXiv is "Team OLMo, Pete Walsh, ..."; entry uses "{Team OLMo} and Walsh, Pete and ... and others" (43 names). Cited as arXiv preprint with a note about the COLM 2025 short version, because the COLM proceedings (OpenReview) could not be fetched to confirm title/authors of that version. |
| 39 | glaive2023functioncalling | VERIFIED | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 ; HF API (createdAt 2023-08-15, license apache-2.0) | Year 2023 from the HF creation date. |
| 40 | yahma2023alpacacleaned | VERIFIED | https://huggingface.co/datasets/yahma/alpaca-cleaned ; HF API (createdAt 2023-03-24) | Year 2023. Original Stanford Alpaca added as `taori2023alpaca` from the README citation block at https://github.com/tatsu-lab/stanford_alpaca (Taori, Gulrajani, Zhang, Dubois, Li, Guestrin, Liang, Hashimoto, 2023). |
| 41 | yu2024metamath | VERIFIED | https://iclr.cc/virtual/2024/poster/18796 ; arXiv API ("ICLR 2024 (Spotlight)") | ICLR 2024. |
| 42 | ji2024pkusaferlhf | VERIFIED | https://aclanthology.org/2025.acl-long.1544/ | **Venue added**: ACL 2025 long papers, pp. 31983-32016, DOI 10.18653/v1/2025.acl-long.1544. Anthology author spellings used ("Josef Dai", "Tianyi Alex Qiu", "Boxun Li"; arXiv has "Juntao Dai", "Tianyi Qiu", "Boxuan Li"). All 13 authors kept. |
| 43 | argilla2024ifevallike | VERIFIED | https://huggingface.co/datasets/argilla/ifeval-like-data ; HF API (createdAt 2024-09-30) | Year 2024. |
| 44 | mangrulkar2022peft | VERIFIED | https://github.com/huggingface/peft (README citation block) | Authors Mangrulkar, Gugger, Debut, Belkada, Paul, Bossan, Tietz; year 2022. |
| 45 | wolf2020transformers | VERIFIED | https://aclanthology.org/2020.emnlp-demos.6/ | EMNLP 2020 System Demonstrations, pp. 38-45, DOI 10.18653/v1/2020.emnlp-demos.6. 22 authors truncated after 12. |
| 46 | alahmari2024repeatability | VERIFIED | Crossref https://api.crossref.org/works/10.1109/ACCESS.2024.3470850 | IEEE Access vol. 12, pp. 153221-153231, 2024. |
| 47 | mugnaini2025consensus | VERIFIED | Crossref https://api.crossref.org/works/10.1109/ACCESS.2025.3601042 | IEEE Access vol. 13, pp. 148192-148202, 2025. |
| 48 | kharinaev2026quantsafety | VERIFIED | Crossref https://api.crossref.org/works/10.1109/ACCESS.2026.3703899 | IEEE Access vol. 14, pp. 96771-96793, 2026. |
| 49 | jaiswal2024truth | VERIFIED | https://proceedings.iclr.cc/paper_files/paper/2024/file/9f09f316a3eaf59d9ced5ffaefe97e0f-Paper-Conference.pdf ; arXiv API ("Accepted to ICLR 2024") | ICLR 2024. |
| 50 | vaswani2017attention | VERIFIED | https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html | NeurIPS (NIPS) 2017, vol. 30. Page numbers are not shown on the proceedings site, so omitted. |
| 51 | compression_safety2025 | VERIFIED | https://arxiv.org/abs/2504.04215 ; https://icml.cc/virtual/2025/49566 | Chosen paper: Chhabra and Khalili, "Towards Understanding and Improving Refusal in Compressed Models via Mechanistic Interpretability", arXiv:2504.04215 (2025) - directly about refusal in pruned/quantized models. A version appeared as a poster at the ICML 2025 Workshop on Actionable Interpretability under a different title ("When Compression Breaks Safety, Interpretability Reveals The Fix"), so the arXiv preprint is cited. A peer-reviewed alternative is included as `xu2024beyondperplexity`: Xu et al., "Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression", Findings of EMNLP 2024, pp. 15359-15396 (https://aclanthology.org/2024.findings-emnlp.901/). Use whichever fits the sentence; delete the other. |
| 52a | frantar2023sparsegpt | VERIFIED | https://proceedings.mlr.press/v202/frantar23a.html | ICML 2023, PMLR 202:10323-10337. |
| 52b | sun2024wanda | VERIFIED | https://proceedings.iclr.cc/paper_files/paper/2024 (listing) ; arXiv API ("ICLR 2024") | **Title**: the paper is "A Simple and Effective Pruning Approach for Large Language Models" ("Wanda" is the method name). ICLR 2024. |

## Corrections to the identifiers you supplied

1. **men2024shortgpt**: published in Findings of **ACL 2025**, not Findings of NAACL 2025.
2. **linearpatch2025**: real title is "A Simple Linear Patch Revives Layer-Pruned Large Language Models"; it is a **NeurIPS 2025** paper, not arXiv-only.
3. **clp2025**: real title is "The Structural Scalpel: Automated Contiguous Layer Pruning for Large Language Models" (Lu et al.).
4. **thangarasa2024selfdata**: published at **MLSys 2025** (Proc. Mach. Learn. Syst. 7).
5. **ji2024pkusaferlhf**: published at **ACL 2025** (long papers).
6. **geng2025jsonschemabench**: current title is "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models" (v1 was "Generating Structured Outputs from Language Models: Benchmark and Studies").
7. **kim2024shortened**: the arXiv title changed in v2; the workshop (ICLR 2024 ME-FoMo) version keeps the "A Simple Depth Pruning" title used in the entry.
8. **sun2024wanda**: title is "A Simple and Effective Pruning Approach for Large Language Models".
9. **gao2024lmeval**: the project's CITATION.bib points to the Dec. 2023 v0.4.0 Zenodo record (10.5281/zenodo.10256836); year 2023 in the entry.
10. **qwen2025qwen25**: arXiv v1 is Dec. 2024; year 2024 in the entry.
11. Full titles filled in for the short-hand items 11, 12, 13, 16, 24, 25, 26 (see table); all arXiv IDs you supplied resolved to the intended papers - none was wrong.

## Things that could not be checked at the source

- OpenReview (forum pages and both API versions) returned a bot-challenge page, and DBLP returned 403; ICLR acceptances were therefore confirmed on proceedings.iclr.cc / iclr.cc instead, and the ME-FoMo workshop page for Shortened LLaMA was confirmed through the arXiv comment and the authors' GitHub README rather than OpenReview.
- The COLM 2025 short version of OLMo 2 could not be inspected (OpenReview); the entry stays an arXiv preprint with a note.
- NeurIPS and ICLR proceedings sites do not expose page numbers, so none are given for those entries (IEEEtran handles this fine).
