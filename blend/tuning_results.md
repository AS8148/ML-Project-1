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
| G oof | logistic Newton, squares + 20 interactions | 0.4414 | 0.0037 | 0.212 | 124s |
| G oof | MLP (128, 64) dropout 0.2, seed 0 | 0.4378 | 0.0043 | 0.212 | 134s |
| G oof | MLP (128, 64) dropout 0.2, seed 1 | 0.4372 | 0.0048 | 0.200 | 124s |
| G oof | MLP (128, 64) dropout 0.2, seed 2 | 0.4381 | 0.0055 | 0.188 | 132s |
| G oof | MLP (128, 64) dropout 0.2, seed 3 | 0.4364 | 0.0036 | 0.212 | 134s |
| G oof | MLP (128, 64) dropout 0.2, seed 4 | 0.4350 | 0.0057 | 0.212 | 130s |
| G oof | MLP (128, 64) dropout 0.2, seed 5 | 0.4372 | 0.0058 | 0.185 | 143s |
| G oof | MLP (128, 64) dropout 0.2, seed 6 | 0.4357 | 0.0055 | 0.210 | 139s |
| G oof | MLP (128, 64) dropout 0.2, seed 7 | 0.4348 | 0.0069 | 0.202 | 126s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 50 trees | 0.4335 | 0.0037 | 0.195 | 1331s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 100 trees | 0.4397 | 0.0048 | 0.210 | 1331s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 150 trees | 0.4417 | 0.0045 | 0.215 | 1332s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 200 trees | 0.4436 | 0.0048 | 0.217 | 1332s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 250 trees | 0.4442 | 0.0053 | 0.217 | 1332s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 300 trees | 0.4446 | 0.0046 | 0.212 | 1332s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 350 trees | 0.4440 | 0.0041 | 0.230 | 1333s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 400 trees | 0.4438 | 0.0046 | 0.220 | 1333s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 450 trees | 0.4444 | 0.0047 | 0.220 | 1333s |
| G gbdt | GBDT max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, 500 trees | 0.4443 | 0.0043 | 0.217 | 1334s |
| G oof | MLP (128, 64) dropout 0.2, seed 8 | 0.4397 | 0.0063 | 0.207 | 126s |
| G oof | MLP (128, 64) dropout 0.2, seed 9 | 0.4360 | 0.0060 | 0.222 | 136s |
| G oof | MLP (128, 64) dropout 0.2, mean of 5 seeds | 0.4428 | 0.0067 | 0.205 | 0s |
| G oof | MLP (128, 64) dropout 0.2, mean of 10 seeds | 0.4434 | 0.0064 | 0.207 | 0s |
| G blend | member logistic | 0.4414 | 0.0037 | 0.212 | 0s |
| G blend | member mlp10 | 0.4434 | 0.0064 | 0.207 | 0s |
| G blend | member gbdt300 | 0.4446 | 0.0046 | 0.212 | 0s |
| G blend | 0.6 mlp10 + 0.4 gbdt300 | 0.4458 | 0.0045 | 0.215 | 0s |
| G blend | 0.4 mlp10 + 0.6 gbdt300 | 0.4458 | 0.0043 | 0.210 | 0s |
| G blend | 0.1 logistic + 0.5 mlp10 + 0.4 gbdt300 | 0.4457 | 0.0046 | 0.212 | 0s |
| G blend | 0.1 logistic + 0.6 mlp10 + 0.3 gbdt300 | 0.4455 | 0.0043 | 0.205 | 0s |
| G blend | 0.5 mlp10 + 0.5 gbdt300 | 0.4455 | 0.0046 | 0.215 | 0s |
| G mlp | MLP hidden=(128, 64), epochs=6, lr=0.001, l2=0.0001, dropout=0.2, lr_schedule=cosine | 0.4400 | 0.0052 | 0.202 | 338s |
| G mlp | MLP hidden=(256, 128), epochs=6, lr=0.001, l2=0.0001, dropout=0.3, lr_schedule=cosine | 0.4404 | 0.0063 | 0.195 | 441s |
| G gbdt | GBDT d6_lr05, 300 trees, seed 1 | 0.4447 | 0.0041 | 0.217 | 914s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 50 trees | 0.4351 | 0.0035 | 0.202 | 1447s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 100 trees | 0.4395 | 0.0041 | 0.217 | 1447s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 150 trees | 0.4406 | 0.0050 | 0.202 | 1448s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 200 trees | 0.4418 | 0.0039 | 0.210 | 1448s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 250 trees | 0.4418 | 0.0035 | 0.215 | 1448s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 300 trees | 0.4420 | 0.0044 | 0.207 | 1448s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 350 trees | 0.4424 | 0.0041 | 0.220 | 1449s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 400 trees | 0.4426 | 0.0049 | 0.217 | 1449s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 450 trees | 0.4425 | 0.0045 | 0.215 | 1449s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 500 trees | 0.4422 | 0.0047 | 0.207 | 1450s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 550 trees | 0.4420 | 0.0045 | 0.207 | 1450s |
| G gbdt | GBDT max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, 600 trees | 0.4419 | 0.0046 | 0.212 | 1450s |
| G mlp | MLP hidden=(512, 256), epochs=8, lr=0.0005, l2=0.0001, dropout=0.4, lr_schedule=cosine | 0.4401 | 0.0060 | 0.185 | 969s |
| G gbdt | GBDT d6_lr05, 300 trees, seed 2 | 0.4442 | 0.0030 | 0.207 | 895s |
| G mlp | MLP (128, 64) cosine 6 epochs, seed 1 | 0.4400 | 0.0060 | 0.205 | 333s |
| G mlp | MLP (128, 64) cosine 6 epochs, seed 2 | 0.4401 | 0.0053 | 0.193 | 318s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 50 trees | 0.4326 | 0.0033 | 0.197 | 1372s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 100 trees | 0.4413 | 0.0034 | 0.215 | 1372s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 150 trees | 0.4432 | 0.0035 | 0.222 | 1373s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 200 trees | 0.4443 | 0.0033 | 0.222 | 1373s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 250 trees | 0.4443 | 0.0041 | 0.227 | 1373s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 300 trees | 0.4442 | 0.0042 | 0.212 | 1374s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 350 trees | 0.4437 | 0.0034 | 0.225 | 1374s |
| G gbdt | GBDT max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, 400 trees | 0.4436 | 0.0039 | 0.217 | 1374s |
| G mlp | MLP (128, 64) cosine 6 epochs, seed 3 | 0.4393 | 0.0061 | 0.202 | 266s |
| G gbdt | GBDT d6_lr05, 300 trees, seed 3 | 0.4448 | 0.0038 | 0.227 | 900s |
| G blend | member logistic | 0.4414 | 0.0037 | 0.212 | 0s |
| G blend | member mlp_x10 | 0.4434 | 0.0064 | 0.207 | 0s |
| G blend | member gbdt_x4 | 0.4459 | 0.0044 | 0.212 | 0s |
| G blend | 0.1 mlp_x10 + 0.9 gbdt_x4 | 0.4465 | 0.0048 | 0.220 | 0s |
| G blend | 0.3 mlp_x10 + 0.7 gbdt_x4 | 0.4464 | 0.0040 | 0.212 | 0s |
| G blend | 0.2 mlp_x10 + 0.8 gbdt_x4 | 0.4464 | 0.0042 | 0.220 | 0s |
| G blend | 0.1 logistic + 0.5 mlp_x10 + 0.4 gbdt_x4 | 0.4463 | 0.0044 | 0.207 | 0s |
| G blend | 0.1 logistic + 0.1 mlp_x10 + 0.8 gbdt_x4 | 0.4463 | 0.0042 | 0.215 | 0s |
| G gbdt | GBDT d6_lr05, 300 trees, seed 4 | 0.4432 | 0.0036 | 0.212 | 597s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 50 trees | 0.4295 | 0.0043 | 0.177 | 1092s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 100 trees | 0.4363 | 0.0048 | 0.202 | 1092s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 150 trees | 0.4401 | 0.0045 | 0.210 | 1093s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 200 trees | 0.4411 | 0.0041 | 0.205 | 1093s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 250 trees | 0.4425 | 0.0040 | 0.207 | 1093s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 300 trees | 0.4432 | 0.0037 | 0.222 | 1093s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 350 trees | 0.4435 | 0.0039 | 0.225 | 1093s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 400 trees | 0.4437 | 0.0039 | 0.212 | 1093s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 450 trees | 0.4443 | 0.0043 | 0.212 | 1094s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 500 trees | 0.4444 | 0.0042 | 0.212 | 1094s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 550 trees | 0.4443 | 0.0044 | 0.215 | 1094s |
| G gbdt | GBDT max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, 600 trees | 0.4449 | 0.0039 | 0.215 | 1094s |
