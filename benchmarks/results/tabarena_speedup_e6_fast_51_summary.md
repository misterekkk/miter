# TabArena lite, 51 tasks: E6_FAST {'preset': 'fast'} vs MITER_SPEEDUP_DEF2

- tasks: 51; wins 22, ties 2, losses 27
- relative error: median +0.065%, mean -0.078%, worst +4.39%
- one-sided Wilcoxon (variant worse): p = 0.4180
- train time (8-fold bag, this machine): total 0.61x, median per task 0.67x

| task | default err | variant err | rel | time ratio |
|---|---:|---:|---:|---:|
| hazelnut-spread-contaminant-detection | 0.026737 | 0.027913 | +4.39% | 0.81x |
| qsar-biodeg | 0.074584 | 0.076279 | +2.27% | 0.75x |
| SDSS17 | 0.077849 | 0.079376 | +1.96% | 0.80x |
| NATICUSdroid | 0.014063 | 0.01427 | +1.47% | 0.49x |
| physiochemical_protein | 3.4251 | 3.4699 | +1.31% | 0.69x |
| wine_quality | 0.60842 | 0.61608 | +1.26% | 0.58x |
| seismic-bumps | 0.19468 | 0.19669 | +1.03% | 0.62x |
| diabetes | 0.16908 | 0.17082 | +1.03% | 0.85x |
| students_dropout_and_academic_success | 0.55455 | 0.55873 | +0.75% | 0.49x |
| miami_housing | 82957 | 83488 | +0.64% | 0.56x |
| houses | 0.21088 | 0.21216 | +0.61% | 0.63x |
| healthcare_insurance_expenses | 4105 | 4128.6 | +0.58% | 0.55x |
| credit_card_clients_default | 0.20753 | 0.20864 | +0.54% | 0.65x |
| Another-Dataset-on-used-Fiat-500 | 690.76 | 693.62 | +0.41% | 0.76x |
| jm1 | 0.24067 | 0.2416 | +0.39% | 0.65x |
| customer_satisfaction_in_airline | 0.0056939 | 0.0057115 | +0.31% | 0.58x |
| taiwanese_bankruptcy_prediction | 0.062777 | 0.062964 | +0.30% | 0.54x |
| QSAR_fish_toxicity | 0.89217 | 0.8945 | +0.26% | 0.98x |
| coil2000_insurance_policies | 0.24484 | 0.24542 | +0.24% | 0.74x |
| heloc | 0.20255 | 0.20304 | +0.24% | 0.58x |
| kddcup09_appetency | 0.18021 | 0.18058 | +0.21% | 0.72x |
| concrete_compressive_strength | 4.4039 | 4.4108 | +0.16% | 0.65x |
| diamonds | 531.22 | 532.05 | +0.16% | 0.74x |
| QSAR-TID-11 | 0.84496 | 0.84583 | +0.10% | 0.45x |
| GiveMeSomeCredit | 0.13101 | 0.13111 | +0.08% | 0.55x |
| churn | 0.068081 | 0.068125 | +0.07% | 0.62x |
| online_shoppers_intention | 0.07309 | 0.073111 | +0.03% | 0.71x |
| splice | 0.091402 | 0.091402 | +0.00% | 1.04x |
| Amazon_employee_access | 0.21391 | 0.21391 | +0.00% | 0.93x |
| superconductivity | 9.3059 | 9.3017 | -0.04% | 0.64x |
| airfoil_self_noise | 1.2126 | 1.212 | -0.05% | 0.82x |
| hiva_agnostic | 0.17392 | 0.1738 | -0.07% | 0.55x |
| Food_Delivery_Time | 7.4421 | 7.4359 | -0.08% | 0.79x |
| blood-transfusion-service-center | 0.29754 | 0.29728 | -0.09% | 0.54x |
| Diabetes130US | 0.32691 | 0.32654 | -0.11% | 0.71x |
| bank-marketing | 0.22495 | 0.2246 | -0.15% | 0.75x |
| HR_Analytics_Job_Change_of_Data_Scientists | 0.19124 | 0.19038 | -0.45% | 1.07x |
| MIC | 0.4987 | 0.49607 | -0.53% | 0.57x |
| Is-this-a-good-customer | 0.29163 | 0.28966 | -0.67% | 0.70x |
| website_phishing | 0.22582 | 0.22429 | -0.68% | 0.55x |
| E-CommereShippingData | 0.2589 | 0.25691 | -0.77% | 0.69x |
| Fitness_Club | 0.19606 | 0.19444 | -0.83% | 0.67x |
| Bioresponse | 0.111 | 0.10989 | -1.00% | 0.58x |
| maternal_health_risk | 0.43515 | 0.42877 | -1.47% | 0.75x |
| Bank_Customer_Churn | 0.1258 | 0.12389 | -1.52% | 1.04x |
| APSFailure | 0.0048635 | 0.0047881 | -1.55% | 0.55x |
| Marketing_Campaign | 0.063098 | 0.061979 | -1.77% | 0.69x |
| credit-g | 0.2438 | 0.23726 | -2.68% | 0.68x |
| polish_companies_bankruptcy | 0.032599 | 0.031665 | -2.87% | 0.56x |
| in_vehicle_coupon_recommendation | 0.17488 | 0.16894 | -3.40% | 0.82x |
| anneal | 0.058522 | 0.056176 | -4.01% | 0.43x |
