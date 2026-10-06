| stage | configuration | CV F1 | std | threshold | time |
|---|---|---|---|---|---|
| A cleaning | NaN<=0.5, no indicators | 0.4259 | 0.0047 | 0.200 | 7s |
| A cleaning | NaN<=0.9 + indicators | 0.4275 | 0.0044 | 0.202 | 16s |
| A cleaning | keep all + indicators | 0.4372 | 0.0053 | 0.199 | 50s |
| A cleaning | keep all + indicators, no clipping | 0.4372 | 0.0057 | 0.201 | 46s |
| A cleaning | keep all + indicators + one-hot state | 0.4376 | 0.0056 | 0.196 | 53s |
| B linear | ridge regression, lambda=1e-05 | 0.4284 | 0.0070 | 0.224 | 4s |
| B linear | ridge regression, lambda=0.001 | 0.4281 | 0.0067 | 0.227 | 3s |
| B linear | ridge regression, lambda=0.1 | 0.4254 | 0.0064 | 0.201 | 4s |
| B linear | logistic GD (300 it, gamma=0.5) | 0.4374 | 0.0054 | 0.205 | 49s |
| B linear | logistic Newton, lambda=0 | 0.4374 | 0.0057 | 0.207 | 77s |
| B linear | logistic Newton, lambda=1e-05 | 0.4379 | 0.0062 | 0.201 | 59s |
| B linear | logistic Newton, lambda=0.0001 | 0.4376 | 0.0056 | 0.196 | 53s |
| B linear | logistic Newton, lambda=0.001 | 0.4376 | 0.0057 | 0.203 | 42s |
| B linear | logistic Newton, lambda=0.01 | 0.4354 | 0.0066 | 0.196 | 42s |
| B linear | logistic Newton, lambda=1e-4, pos_weight=2 | 0.4372 | 0.0062 | 0.334 | 53s |
| B linear | logistic Newton, lambda=1e-4, pos_weight=5 | 0.4370 | 0.0053 | 0.560 | 59s |
| B linear | logistic Newton, lambda=1e-4, pos_weight=10 | 0.4367 | 0.0051 | 0.722 | 62s |
| C features | logistic Newton lambda=0.0001, squares | 0.4384 | 0.0051 | 0.209 | 61s |
| C features | logistic Newton lambda=0.001, squares | 0.4390 | 0.0045 | 0.200 | 48s |
| C features | logistic Newton lambda=0.0001, squares + 20 interactions | 0.4418 | 0.0039 | 0.216 | 90s |
| C features | logistic Newton lambda=0.001, squares + 20 interactions | 0.4415 | 0.0045 | 0.217 | 80s |
| C features | logistic Newton lambda=0.0001, squares + 40 interactions | 0.4399 | 0.0040 | 0.218 | 230s |
| C features | logistic Newton lambda=0.001, squares + 40 interactions | 0.4408 | 0.0040 | 0.225 | 203s |
| D MLP | MLP hidden=32, epochs=4, lr=0.001, l2=0.0001 | 0.4321 | 0.0034 | 0.234 | 30s |
| D MLP | MLP hidden=64, epochs=4, lr=0.001, l2=0.0001 | 0.4321 | 0.0084 | 0.224 | 34s |
| D MLP | MLP hidden=64, epochs=8, lr=0.0005, l2=0.0001 | 0.4232 | 0.0047 | 0.263 | 67s |
| D MLP | MLP hidden=128, epochs=4, lr=0.0005, l2=0.001 | 0.4336 | 0.0088 | 0.230 | 45s |
| D MLP | MLP hidden=64, epochs=4, lr=0.001, l2=0.001 | 0.4338 | 0.0081 | 0.211 | 35s |
| D MLP | MLP hidden=64, epochs=2, lr=0.001, l2=0.001 | 0.4334 | 0.0056 | 0.205 | 15s |
| D MLP | MLP hidden=64, epochs=4, lr=0.001, l2=0.003 | 0.4315 | 0.0090 | 0.210 | 33s |
| D MLP | MLP hidden=128, epochs=3, lr=0.001, l2=0.003 | 0.4293 | 0.0048 | 0.228 | 34s |
| E ensemble | MLP x5 seeds | 0.4401 | 0.0044 | 0.217 | 178s |
| E ensemble | 0.3 * logistic(expanded) + 0.7 * MLP x5 | 0.4412 | 0.0057 | 0.214 | 271s |
| E ensemble | 0.5 * logistic(expanded) + 0.5 * MLP x5 | 0.4414 | 0.0053 | 0.212 | 265s |
| E ensemble | 0.7 * logistic(expanded) + 0.3 * MLP x5 | 0.4421 | 0.0038 | 0.211 | 287s |
| F blend | logistic Newton, squares + 20 interactions | 0.4418 | 0.0039 | 0.216 | 91s |
| F MLP | MLP hidden=64, epochs=4, lr=0.001, l2=0.001 | 0.4338 | 0.0081 | 0.211 | 38s |
| F MLP | MLP hidden=64, epochs=4, lr=0.001, l2=0.0001, dropout=0.2 | 0.4345 | 0.0043 | 0.223 | 38s |
| F MLP | MLP hidden=256, epochs=4, lr=0.001, l2=0.0001, dropout=0.4 | 0.4358 | 0.0050 | 0.239 | 69s |
| F MLP | MLP hidden=(128, 64), epochs=4, lr=0.001, l2=0.0001, dropout=0.2 | 0.4371 | 0.0037 | 0.231 | 50s |
| F MLP | MLP hidden=(256, 128), epochs=5, lr=0.0005, l2=0.0001, dropout=0.3 | 0.4353 | 0.0063 | 0.207 | 122s |
| F MLP | MLP x5 seeds, hidden=(128, 64), epochs=4, lr=0.001, l2=0.0001, dropout=0.2 | 0.4425 | 0.0068 | 0.213 | 302s |
| F blend | 0.2 * logistic(expanded) + 0.8 * MLP x5 (hidden=(128, 64), epochs=4, lr=0.001, l2=0.0001, dropout=0.2) | 0.4425 | 0.0064 | 0.216 | 0s |
| F blend | 0.35 * logistic(expanded) + 0.65 * MLP x5 (hidden=(128, 64), epochs=4, lr=0.001, l2=0.0001, dropout=0.2) | 0.4424 | 0.0056 | 0.217 | 0s |
| F blend | 0.5 * logistic(expanded) + 0.5 * MLP x5 (hidden=(128, 64), epochs=4, lr=0.001, l2=0.0001, dropout=0.2) | 0.4430 | 0.0052 | 0.218 | 0s |
| F blend | 0.65 * logistic(expanded) + 0.35 * MLP x5 (hidden=(128, 64), epochs=4, lr=0.001, l2=0.0001, dropout=0.2) | 0.4427 | 0.0050 | 0.214 | 0s |
| F blend | 0.8 * logistic(expanded) + 0.2 * MLP x5 (hidden=(128, 64), epochs=4, lr=0.001, l2=0.0001, dropout=0.2) | 0.4416 | 0.0049 | 0.217 | 0s |
