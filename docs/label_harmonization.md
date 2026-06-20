# Label harmonization

The initial canonical order is `AF`, `RBBB`, `LBBB`, `1dAVB`, `SB`, `ST`. Output columns and
metrics must retain this order. Mapping is presence-based: any listed native code activates the
canonical label, and labels outside this task are ignored rather than treated as diagnoses.

| Canonical | PTB-XL | Chapman | SPH | CODE-15% |
|---|---|---|---|---|
| AF | AFIB | AFIB | 50, 50+346, 50+347 | AF |
| RBBB | CRBBB, IRBBB | RBBB | 106 | RBBB |
| LBBB | CLBBB, ILBBB | LBBB | 104 | LBBB |
| 1dAVB | 1AVB | 1AVB | 82 | 1dAVb |
| SB | SBRAD | SB | 22 | SB |
| ST | STACH | ST | 21 | ST |

RBBB and LBBB merge complete and incomplete forms where separate native codes exist. This
choice increases cross-source coverage but may collapse clinically meaningful severity. SPH
`1dAVB` is approximated by the prolonged-PR code and must be treated as a mapping limitation.
`ST` always denotes **sinus tachycardia**, never ST elevation/depression or a general ST-segment
abnormality.

Before benchmark claims are made, mappings must be checked against the exact dataset release,
official code definitions, multilabel parsing, and a sample audit. Changes require a versioned
mapping and regenerated processed indexes.
