# M3 parity re-check after the axis-binning fix (miter C++ vs histree T7 Q16r_int)

- Hardware: `arm64` (`macOS-26.6.2-arm64-arm-64bit-Mach-O`); miter `0.1.0.dev0`; built from `876f133` (axis-binner fix `876f133`)
- Protocol: as `bench_m3_oblique.py` Part B (5 folds, seed = fold, 1 thread). T7 = cached `clean|Q16r_int` (reproduced exactly by a live run, see `m3_parity_brazil_pol_summary.md`).
- Mean nRMSE over 5 folds; negative = miter lower error. Last column: Part B before the fix.

| dataset | miter nRMSE | histree T7 | miter vs T7 | before fix |
|---|---|---|---|---|
| abalone | 0.6657 | 0.6640 | +0.25% | -0.12% |
| wine_quality | 0.7014 | 0.7058 | -0.62% | -0.80% |
| cpu_act | 0.1177 | 0.1175 | +0.11% | -2.87% |
| yprop_4_1 | 0.9537 | 0.9559 | -0.23% | +0.09% |
| sulfur | 0.3681 | 0.3808 | -3.34% | -3.31% |
| Brazilian_houses | 0.0457 | 0.0447 | +2.42% | +13.09% |
| Ailerons | 0.3876 | 0.3880 | -0.09% | -0.09% |
| MiamiHousing2016 | 0.2599 | 0.2612 | -0.49% | -0.61% |
| pol | 0.0991 | 0.0965 | +2.67% | +3.37% |
| elevators | 0.2911 | 0.2903 | +0.26% | -0.08% |
| Bike_Sharing_Demand | 0.5398 | 0.5388 | +0.18% | +0.10% |
| houses | 0.3687 | 0.3680 | +0.19% | +0.09% |
| superconduct | 0.2664 | 0.2666 | -0.10% | -0.14% |
| house_sales | 0.3287 | 0.3277 | +0.30% | -0.41% |
| house_16H | 0.6731 | 0.6702 | +0.44% | -1.61% |
| diamonds | 0.2277 | 0.2275 | +0.07% | +0.11% |

- **After fix (16 reg):** mean +0.13%, **median +0.14%**, wins 6/16, Wilcoxon p = 0.7057, max |diff| 3.34%
- Before fix (Part B, same protocol): mean +0.43%, median -0.10%
