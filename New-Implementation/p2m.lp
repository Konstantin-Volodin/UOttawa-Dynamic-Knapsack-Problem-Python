\ Model Master problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 58.21197000000001 sa_2 - 29.103 sa_313 + 29.7 sa_355 - 12.394788 sa_369
   + 26.73 sa_372 - 17.265586 sa_373 - 58.21197000000001 sa_374
   + 29.7 sa_376 + 0.356409 sa_386 - 29.103 sa_389 - 28.987182 sa_390
   - 29.103 sa_393 - 29.103 sa_395 + 8.91 sa_396 - 24.252898 sa_397
   + 29.7 sa_401 - 1.741399 sa_403 + 0.3 sa_406 - 17.463591 sa_407
   + 29.7 sa_408 - 58.21197000000001 sa_409 - 8.5509 sa_410 - 8.5509 sa_412
   - 1.781 sa_413 - 26.017182 sa_415 + 0.3 sa_416 - 28.09499 sa_417
   - 31.995788 sa_418 - 31.995788 sa_420 + 13.7907 sa_421 - 10.6711 sa_422
   - 21.322399 sa_423 - 3.5318 sa_424 + 2.97 sa_426 - 10.6711 sa_427
   - 42.687783 sa_428 + 1.1099 sa_429 - 8.5509 sa_430 - 10.6511 sa_431
   - 3.722394000000001 sa_432 - 58.21197000000001 sa_433 + 0.3 sa_434
   - 3.722394000000001 sa_435 - 58.21197000000001 sa_436
   - 2.790798000000001 sa_437 - 58.21197000000001 sa_438
   - 3.722394000000001 sa_439 + 29.7 sa_440
Subject To
 con_beta_0: 0.01 sa_2 + 0.01 sa_313 + 0.01 sa_355 + 0.01 sa_369
   + 0.01 sa_372 + 0.01 sa_373 + 0.01 sa_374 + 0.01 sa_376 + 0.01 sa_386
   + 0.01 sa_389 + 0.01 sa_390 + 0.01 sa_391 + 0.01 sa_393 + 0.01 sa_395
   + 0.01 sa_396 + 0.01 sa_397 + 0.01 sa_401 + 0.01 sa_403 + 0.01 sa_406
   + 0.01 sa_407 + 0.01 sa_408 + 0.01 sa_409 + 0.01 sa_410 + 0.01 sa_411
   + 0.01 sa_412 + 0.01 sa_413 + 0.01 sa_415 + 0.01 sa_416 + 0.01 sa_417
   + 0.01 sa_418 + 0.01 sa_420 + 0.01 sa_421 + 0.01 sa_422 + 0.01 sa_423
   + 0.01 sa_424 + 0.01 sa_425 + 0.01 sa_426 + 0.01 sa_427 + 0.01 sa_428
   + 0.01 sa_429 + 0.01 sa_430 + 0.01 sa_431 + 0.01 sa_432 + 0.01 sa_433
   + 0.01 sa_434 + 0.01 sa_435 + 0.01 sa_436 + 0.01 sa_437 + 0.01 sa_438
   + 0.01 sa_439 + 0.01 sa_440 = 1
 con_beta_ul_nco[PPE1]: 60 sa_2 + 60 sa_313 + 60 sa_355 + 60 sa_369
   + 60 sa_372 + 60 sa_373 + 60 sa_374 + 60 sa_376 + 60 sa_386 + 60 sa_389
   + 60 sa_390 + 60 sa_391 + 60 sa_393 + 60 sa_395 + 60 sa_396 + 60 sa_397
   + 60 sa_401 + 60 sa_403 + 60 sa_406 + 60 sa_407 + 60 sa_408 + 60 sa_409
   + 60 sa_410 + 60 sa_411 + 60 sa_412 + 60 sa_413 + 60 sa_415 + 60 sa_416
   + 60 sa_417 + 60 sa_418 + 60 sa_420 + 60 sa_421 + 60 sa_422 + 60 sa_423
   + 60 sa_424 + 60 sa_425 + 60 sa_426 + 60 sa_427 + 60 sa_428 + 60 sa_429
   + 60 sa_430 + 60 sa_431 + 60 sa_432 + 60 sa_433 + 60 sa_434 + 60 sa_435
   + 60 sa_436 + 60 sa_437 + 60 sa_438 + 60 sa_439 + 60 sa_440 >= 0
 con_beta_pw_0dkc[0,C1,P1,CPU1]: - 8.91 sa_2 - 8.91 sa_313 - 8.91 sa_355
   - 8.91 sa_369 - 8.91 sa_372 - 8.91 sa_373 - 8.91 sa_374 + 21.09 sa_376
   - 8.91 sa_386 - 8.91 sa_389 - 8.91 sa_390 - 8.91 sa_391 - 8.91 sa_393
   - 8.91 sa_395 - 8.91 sa_396 - 8.91 sa_397 - 8.91 sa_401 - 8.91 sa_403
   - 8.91 sa_406 - 8.91 sa_407 - 8.91 sa_408 - 8.91 sa_409 - 8.91 sa_410
   - 8.91 sa_411 - 8.91 sa_412 - 8.91 sa_413 - 8.91 sa_415 - 8.91 sa_416
   - 8.91 sa_417 - 8.91 sa_418 - 8.91 sa_420 - 8.91 sa_421 - 8.91 sa_422
   - 8.91 sa_423 - 8.91 sa_424 - 8.91 sa_425 - 6.91 sa_426 - 8.91 sa_427
   - 8.91 sa_428 - 6.91 sa_429 - 8.91 sa_430 - 8.91 sa_431 - 8.91 sa_432
   - 8.91 sa_433 - 8.91 sa_434 - 8.91 sa_435 - 8.91 sa_436 - 8.91 sa_437
   - 8.91 sa_438 - 8.91 sa_439 - 8.91 sa_440 >= 9
 con_beta_pw_0dkc[0,C2,P1,CPU1]: >= 0
 con_beta_pw_1TLdkc[1,C1,P1,CPU1]: 30 sa_408 >= 8
 con_beta_pw_1TLdkc[1,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[2,C1,P1,CPU1]: 18 sa_372 + 6 sa_390 + 30 sa_440 >= 8
 con_beta_pw_TLMdkc[2,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[3,C1,P1,CPU1]: 18 sa_386 + 30 sa_401 >= 7
 con_beta_pw_TLMdkc[3,C2,P1,CPU1]: sa_369 + 10 sa_373 + 9 sa_396
   + 10 sa_403 + 8 sa_413 + 9 sa_415 + 7 sa_421 + 8 sa_432 + 8 sa_435
   + 8 sa_439 >= 4
 con_beta_pw_Mdkc[4,C1,P1,CPU1]: 30 sa_355 >= 7
 con_beta_pw_Mdkc[4,C2,P1,CPU1]: 10 sa_369 + 9 sa_372 + sa_417 + sa_418
   + sa_420 + sa_426 + sa_437 >= 4
 con_beta_ps_t0dkc[1,0,C1,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[1,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[2,0,C1,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[2,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[3,0,C1,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[3,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[1,1,C1,P1,CPU1]: 2 sa_403 + 30 sa_411 + 30 sa_421
   + 2 sa_422 + 2 sa_424 >= 1
 con_beta_ps_t1TLdkc[1,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[2,1,C1,P1,CPU1]: 30 sa_406 + 10 sa_423 + 2 sa_430
   >= 1
 con_beta_ps_t1TLdkc[2,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[3,1,C1,P1,CPU1]: 30 sa_395 >= 1
 con_beta_ps_t1TLdkc[3,1,C2,P1,CPU1]: sa_397 + 10 sa_403 + sa_422
   + 7 sa_423 + 7 sa_424 + sa_427 >= 1
 con_beta_ps_tTLMdkc[1,2,C1,P1,CPU1]: 18 sa_407 + 6 sa_413 - 8.91 sa_423
   + 30 sa_425 + 2 sa_427 - 1.782 sa_430 >= 1
 con_beta_ps_tTLMdkc[1,2,C2,P1,CPU1]: - 0.99 sa_423 + 10 sa_424 + 10 sa_426
   + 10 sa_427 - 0.198 sa_430 + 10 sa_432 + 10 sa_435 >= 0
 con_beta_ps_tTLMdkc[1,3,C1,P1,CPU1]: 30 sa_391 - 8.91 sa_422 - 8.91 sa_427
   >= 1
 con_beta_ps_tTLMdkc[1,3,C2,P1,CPU1]: - 0.99 sa_422 - 6.93 sa_424
   - 6.93 sa_425 + 10 sa_426 - 0.99 sa_427 + 10 sa_429 - 5.94 sa_430
   + 10 sa_431 + 10 sa_437 + 3.07 sa_439 >= 0
 con_beta_ps_tTLMdkc[2,2,C1,P1,CPU1]: 18 sa_410 + 2 sa_417 + 10 sa_422
   + 10 sa_427 >= 1
 con_beta_ps_tTLMdkc[2,2,C2,P1,CPU1]: - 6.93 sa_423 - 6.93 sa_424
   + 7 sa_425 + 10 sa_429 + 6 sa_430 + 10 sa_437 + 7 sa_439 >= 0
 con_beta_ps_tTLMdkc[2,3,C1,P1,CPU1]: 30 sa_416 + 10 sa_433 >= 1
 con_beta_ps_tTLMdkc[2,3,C2,P1,CPU1]: - 6.93 sa_425 - 6.93 sa_426
   - 6.93 sa_428 + 7 sa_432 + 6 sa_434 + 7 sa_437 + 7 sa_438 >= 0
 con_beta_ps_tTLMdkc[3,2,C1,P1,CPU1]: 30 sa_313 >= 1
 con_beta_ps_tTLMdkc[3,2,C2,P1,CPU1]: 9 sa_410 + 9 sa_412 + 10 sa_413
   + 10 sa_422 + 7 sa_425 + 7 sa_426 + 10 sa_427 + 7 sa_428 + sa_431 >= 1
 con_beta_ps_tTLMdkc[3,3,C1,P1,CPU1]: 30 sa_389 + sa_428 >= 1
 con_beta_ps_tTLMdkc[3,3,C2,P1,CPU1]: 10 sa_397 + 10 sa_418 + 10 sa_420
   + 7 sa_421 + 10 sa_423 + 8 sa_424 + 8 sa_428 + 7 sa_430 + 7 sa_431
   + 6 sa_433 + 7 sa_440 >= 1
 con_beta_ps_tMdkc[1,4,C1,P1,CPU1]: 18 sa_396 - 8.91 sa_428 - 8.91 sa_429
   - 8.91 sa_431 - 8.91 sa_433 - 8.91 sa_435 - 8.91 sa_436 >= 1
 con_beta_ps_tMdkc[1,4,C2,P1,CPU1]: - 6.93 sa_421 + 10 sa_422 - 0.99 sa_428
   - 0.99 sa_429 - 0.99 sa_431 - 6.93 sa_432 - 0.99 sa_433 - 6.93 sa_434
   - 0.99 sa_435 - 0.99 sa_436 - 6.93 sa_437 - 6.93 sa_438 >= 0
 con_beta_ps_tMdkc[2,4,C1,P1,CPU1]: 18 sa_412 + 2 sa_423 + 10 sa_428
   + 10 sa_429 + 18 sa_430 + 12 sa_431 + 30 sa_434 + 10 sa_435 + 10 sa_436
   >= 1
 con_beta_ps_tMdkc[2,4,C2,P1,CPU1]: - 6.93 sa_421 - 6.93 sa_422
   - 6.93 sa_427 - 6.93 sa_429 - 6.93 sa_430 - 6.93 sa_431 - 6.93 sa_433
   + sa_434 - 6.93 sa_440 >= 0
 con_beta_ps_tMdkc[3,4,C1,P1,CPU1]: 30 sa_393 >= 1
 con_beta_ps_tMdkc[3,4,C2,P1,CPU1]: 10 sa_397 + 10 sa_417 + 7 sa_422
   + 10 sa_423 + 10 sa_424 + 7 sa_427 + sa_428 + 8 sa_429 + 9 sa_430
   + 10 sa_431 + sa_433 >= 1
 con_beta_ps_Tmdkc[4,0,C1,P1,CPU1]: 2 sa_418 + 2 sa_420 + 6 sa_435 >= 0
 con_beta_ps_Tmdkc[4,0,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,1,C1,P1,CPU1]: 30 sa_409 >= 1
 con_beta_ps_Tmdkc[4,1,C2,P1,CPU1]: 8 sa_373 + 9 sa_407 + sa_423 >= 1
 con_beta_ps_Tmdkc[4,2,C1,P1,CPU1]: 30 sa_2 + 30 sa_438 >= 1
 con_beta_ps_Tmdkc[4,2,C2,P1,CPU1]: 10 sa_369 + 8 sa_390 >= 1
 con_beta_ps_Tmdkc[4,3,C1,P1,CPU1]: 2 sa_369 + 6 sa_373 + 30 sa_374
   + 6 sa_432 + 30 sa_433 + 2 sa_437 + 6 sa_439 >= 1
 con_beta_ps_Tmdkc[4,3,C2,P1,CPU1]: 9 sa_386 >= 1
 con_beta_ps_Tmdkc[4,4,C1,P1,CPU1]: 2 sa_397 + 18 sa_415 + 17 sa_428
   + 30 sa_436 >= 1
 con_beta_ps_Tmdkc[4,4,C2,P1,CPU1]: 10 sa_390 + sa_403 + 10 sa_417
   + 10 sa_418 + 10 sa_420 >= 1
Bounds
End
