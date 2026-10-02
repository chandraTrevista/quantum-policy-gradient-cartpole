| Policy | Parameters | Final return, IQM [95% CI] | Training return, IQM [95% CI] | Runs solved | Median episodes to solve |
|---|---:|---:|---:|---:|---:|
| `quantum` | 69 | 459 [358, 492] | 351 [319, 377] | 10/10 | 422 |
| `quantum_fixed_input_scaling` | 49 | 333 [188, 465] | 304 [278, 330] | 9/10 | 584 |
| `quantum_fixed_output_scale` | 68 | 332 [303, 356] | 246 [227, 265] | 0/10 | - |
| `quantum_1_layer` | 21 | 389 [243, 482] | 266 [175, 318] | 5/10 | 577 |
| `quantum_3_layers` | 45 | 421 [267, 490] | 348 [269, 382] | 7/10 | 380 |
| `mlp_matched` | 72 | 499 [493, 500] | 406 [385, 413] | 10/10 | 362 |
| `mlp_large` | 4,610 | 493 [449, 500] | 406 [383, 422] | 10/10 | 322 |

| Shots per decision | Evaluation return, IQM [95% CI] |
|---:|---:|
| 1 | 86 [64, 128] |
| 4 | 185 [129, 276] |
| 16 | 424 [301, 461] |
| 64 | 456 [325, 499] |
| 256 | 464 [329, 499] |
| 1024 | 463 [325, 500] |
| exact | 466 [334, 498] |
