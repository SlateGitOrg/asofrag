# Ablation: as-of retrieval vs naive recency-boosted retrieval

60 questions, 5 context blocks per question.

| mode | recall@3 | out-of-force citation rate | quote groundedness | answer accuracy | abstention accuracy |
|---|---|---|---|---|---|
| naive | 100.0% | 60.0% | 100.0% | 50.0% | 40.0% |
| as-of | 100.0% | 0.0% | 100.0% | 100.0% | 98.3% |

| mode | abstention precision | abstention recall |
|---|---|---|
| naive | n/a | 0.0% |
| as-of | 100.0% | 91.7% |

| count | naive | as-of |
|---|---|---|
| before-change questions citing the newer (out-of-force) version | 24 of 24 | 0 of 24 |
| abstentions on the 12 abstain questions | 0 of 12 | 11 of 12 |
| abstentions on the 48 answerable questions | 0 of 48 | 0 of 48 |
| unchanged-clause questions with gold in top 3 | 16 of 16 | 16 of 16 |

## Fault plan (as-of mode, first attempts only)

7 faults injected; codes caught on the first attempt: CITATION_NOT_RETRIEVED x2, CITATION_OUT_OF_FORCE x2, QUOTE_NOT_FOUND x3.

| question | fault | expected code | codes seen | attempts | correct after re-ask |
|---|---|---|---|---|---|
| Q18 | superseded_citation | CITATION_OUT_OF_FORCE | CITATION_OUT_OF_FORCE | 2 | yes |
| Q25 | superseded_citation | CITATION_OUT_OF_FORCE | CITATION_OUT_OF_FORCE | 2 | yes |
| Q03 | fabricated_quote | QUOTE_NOT_FOUND | QUOTE_NOT_FOUND | 2 | yes |
| Q08 | fabricated_quote | QUOTE_NOT_FOUND | QUOTE_NOT_FOUND | 2 | yes |
| Q09 | fabricated_quote | QUOTE_NOT_FOUND | QUOTE_NOT_FOUND | 2 | yes |
| Q10 | unretrieved_citation | CITATION_NOT_RETRIEVED | CITATION_NOT_RETRIEVED | 2 | yes |
| Q11 | unretrieved_citation | CITATION_NOT_RETRIEVED | CITATION_NOT_RETRIEVED | 2 | yes |
