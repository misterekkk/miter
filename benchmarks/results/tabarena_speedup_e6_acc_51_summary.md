# TabArena lite, 51 tasks: E6_ACC {'preset': 'accurate'} vs MITER_SPEEDUP_DEF2

- tasks: 51; wins 27, ties 2, losses 22
- relative error: median -0.004%, mean -0.461%, worst +1.31%
- one-sided Wilcoxon (variant worse): p = 0.8361
- train time (8-fold bag, this machine): total 1.16x, median per task 1.22x

| task | default err | variant err | rel | time ratio |
|---|---:|---:|---:|---:|
| airfoil_self_noise | 1.2126 | 1.2285 | +1.31% | 1.34x |
| miami_housing | 82957 | 83783 | +1.00% | 0.95x |
| Bank_Customer_Churn | 0.1258 | 0.12684 | +0.83% | 1.31x |
| seismic-bumps | 0.19468 | 0.19625 | +0.81% | 1.98x |
| kddcup09_appetency | 0.18021 | 0.1816 | +0.77% | 1.21x |
| coil2000_insurance_policies | 0.24484 | 0.24646 | +0.66% | 0.89x |
| in_vehicle_coupon_recommendation | 0.17488 | 0.17592 | +0.59% | 1.33x |
| Is-this-a-good-customer | 0.29163 | 0.2933 | +0.57% | 1.63x |
| Fitness_Club | 0.19606 | 0.19705 | +0.50% | 1.22x |
| credit-g | 0.2438 | 0.24466 | +0.35% | 1.22x |
| website_phishing | 0.22582 | 0.22658 | +0.34% | 1.10x |
| wine_quality | 0.60842 | 0.61033 | +0.31% | 0.96x |
| anneal | 0.058522 | 0.058698 | +0.30% | 1.33x |
| customer_satisfaction_in_airline | 0.0056939 | 0.0057108 | +0.30% | 0.98x |
| online_shoppers_intention | 0.07309 | 0.073298 | +0.28% | 1.52x |
| QSAR-TID-11 | 0.84496 | 0.84655 | +0.19% | 0.99x |
| diamonds | 531.22 | 532.15 | +0.18% | 1.36x |
| QSAR_fish_toxicity | 0.89217 | 0.89327 | +0.12% | 1.34x |
| jm1 | 0.24067 | 0.2409 | +0.10% | 1.23x |
| Diabetes130US | 0.32691 | 0.32706 | +0.05% | 0.97x |
| E-CommereShippingData | 0.2589 | 0.25897 | +0.02% | 1.42x |
| Food_Delivery_Time | 7.4421 | 7.4434 | +0.02% | 1.03x |
| splice | 0.091402 | 0.091402 | +0.00% | 1.05x |
| Amazon_employee_access | 0.21391 | 0.21391 | +0.00% | 1.10x |
| GiveMeSomeCredit | 0.13101 | 0.13101 | -0.00% | 1.10x |
| healthcare_insurance_expenses | 4105 | 4104.8 | -0.00% | 1.18x |
| HR_Analytics_Job_Change_of_Data_Scientists | 0.19124 | 0.19123 | -0.01% | 1.28x |
| bank-marketing | 0.22495 | 0.22493 | -0.01% | 1.29x |
| hiva_agnostic | 0.17392 | 0.17387 | -0.03% | 1.14x |
| MIC | 0.4987 | 0.49845 | -0.05% | 0.97x |
| superconductivity | 9.3059 | 9.2973 | -0.09% | 1.22x |
| houses | 0.21088 | 0.21067 | -0.10% | 1.22x |
| credit_card_clients_default | 0.20753 | 0.20708 | -0.22% | 1.47x |
| students_dropout_and_academic_success | 0.55455 | 0.55293 | -0.29% | 1.24x |
| Bioresponse | 0.111 | 0.11065 | -0.31% | 1.08x |
| diabetes | 0.16908 | 0.16834 | -0.44% | 1.17x |
| heloc | 0.20255 | 0.20159 | -0.47% | 1.33x |
| concrete_compressive_strength | 4.4039 | 4.3818 | -0.50% | 1.82x |
| NATICUSdroid | 0.014063 | 0.013987 | -0.54% | 0.85x |
| blood-transfusion-service-center | 0.29754 | 0.29553 | -0.68% | 1.15x |
| Another-Dataset-on-used-Fiat-500 | 690.76 | 685.99 | -0.69% | 2.26x |
| churn | 0.068081 | 0.067557 | -0.77% | 1.33x |
| Marketing_Campaign | 0.063098 | 0.06256 | -0.85% | 2.01x |
| taiwanese_bankruptcy_prediction | 0.062777 | 0.062235 | -0.86% | 1.01x |
| physiochemical_protein | 3.4251 | 3.3922 | -0.96% | 1.40x |
| SDSS17 | 0.077849 | 0.077019 | -1.07% | 1.49x |
| qsar-biodeg | 0.074584 | 0.07307 | -2.03% | 1.09x |
| maternal_health_risk | 0.43515 | 0.42095 | -3.26% | 1.62x |
| polish_companies_bankruptcy | 0.032599 | 0.03126 | -4.11% | 1.38x |
| hazelnut-spread-contaminant-detection | 0.026737 | 0.024906 | -6.85% | 1.40x |
| APSFailure | 0.0048635 | 0.0044788 | -7.91% | 1.15x |
