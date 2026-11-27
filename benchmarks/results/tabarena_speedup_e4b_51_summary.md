# TabArena lite, 51 tasks: E4B {'oblique_gain_margin': 0.05} vs MITER_SPEEDUP_DEF2

- tasks: 51; wins 27, ties 2, losses 22
- relative error: median -0.004%, mean -0.065%, worst +1.69%
- one-sided Wilcoxon (variant worse): p = 0.6461
- train time (8-fold bag, this machine): total 0.96x, median per task 0.99x

| task | default err | variant err | rel | time ratio |
|---|---:|---:|---:|---:|
| qsar-biodeg | 0.074584 | 0.075847 | +1.69% | 0.99x |
| Is-this-a-good-customer | 0.29163 | 0.29481 | +1.09% | 1.04x |
| churn | 0.068081 | 0.06878 | +1.03% | 1.08x |
| wine_quality | 0.60842 | 0.61453 | +1.00% | 0.78x |
| blood-transfusion-service-center | 0.29754 | 0.2993 | +0.59% | 1.10x |
| seismic-bumps | 0.19468 | 0.19577 | +0.56% | 0.95x |
| healthcare_insurance_expenses | 4105 | 4125.1 | +0.49% | 0.86x |
| taiwanese_bankruptcy_prediction | 0.062777 | 0.063045 | +0.43% | 0.96x |
| coil2000_insurance_policies | 0.24484 | 0.24569 | +0.35% | 0.83x |
| website_phishing | 0.22582 | 0.22651 | +0.31% | 0.95x |
| superconductivity | 9.3059 | 9.3282 | +0.24% | 0.93x |
| E-CommereShippingData | 0.2589 | 0.25935 | +0.17% | 1.52x |
| customer_satisfaction_in_airline | 0.0056939 | 0.0057029 | +0.16% | 0.95x |
| credit_card_clients_default | 0.20753 | 0.20785 | +0.16% | 0.92x |
| anneal | 0.058522 | 0.058609 | +0.15% | 0.96x |
| MIC | 0.4987 | 0.49919 | +0.10% | 0.90x |
| bank-marketing | 0.22495 | 0.22516 | +0.09% | 1.03x |
| kddcup09_appetency | 0.18021 | 0.18037 | +0.09% | 1.04x |
| credit-g | 0.2438 | 0.24393 | +0.05% | 1.12x |
| Diabetes130US | 0.32691 | 0.32697 | +0.02% | 1.02x |
| QSAR-TID-11 | 0.84496 | 0.84504 | +0.01% | 0.88x |
| GiveMeSomeCredit | 0.13101 | 0.13101 | +0.00% | 0.98x |
| Amazon_employee_access | 0.21391 | 0.21391 | +0.00% | 1.00x |
| splice | 0.091402 | 0.091402 | +0.00% | 0.94x |
| heloc | 0.20255 | 0.20255 | -0.00% | 0.95x |
| Food_Delivery_Time | 7.4421 | 7.4417 | -0.00% | 0.86x |
| SDSS17 | 0.077849 | 0.077839 | -0.01% | 0.98x |
| hiva_agnostic | 0.17392 | 0.17388 | -0.02% | 1.07x |
| Fitness_Club | 0.19606 | 0.19599 | -0.04% | 1.08x |
| Bioresponse | 0.111 | 0.11095 | -0.04% | 0.87x |
| online_shoppers_intention | 0.07309 | 0.073056 | -0.05% | 1.03x |
| students_dropout_and_academic_success | 0.55455 | 0.55417 | -0.07% | 1.02x |
| diabetes | 0.16908 | 0.16894 | -0.08% | 0.85x |
| miami_housing | 82957 | 82883 | -0.09% | 0.87x |
| QSAR_fish_toxicity | 0.89217 | 0.89101 | -0.13% | 1.13x |
| in_vehicle_coupon_recommendation | 0.17488 | 0.17457 | -0.18% | 0.91x |
| physiochemical_protein | 3.4251 | 3.4189 | -0.18% | 1.04x |
| hazelnut-spread-contaminant-detection | 0.026737 | 0.026675 | -0.23% | 1.04x |
| Another-Dataset-on-used-Fiat-500 | 690.76 | 689.09 | -0.24% | 1.28x |
| diamonds | 531.22 | 529.91 | -0.25% | 1.17x |
| NATICUSdroid | 0.014063 | 0.014023 | -0.28% | 0.95x |
| polish_companies_bankruptcy | 0.032599 | 0.032499 | -0.31% | 0.92x |
| Bank_Customer_Churn | 0.1258 | 0.12539 | -0.33% | 1.22x |
| airfoil_self_noise | 1.2126 | 1.2083 | -0.35% | 1.03x |
| HR_Analytics_Job_Change_of_Data_Scientists | 0.19124 | 0.19054 | -0.36% | 1.14x |
| houses | 0.21088 | 0.21008 | -0.38% | 1.02x |
| jm1 | 0.24067 | 0.23877 | -0.79% | 1.07x |
| concrete_compressive_strength | 4.4039 | 4.3676 | -0.82% | 0.94x |
| maternal_health_risk | 0.43515 | 0.42682 | -1.91% | 1.08x |
| APSFailure | 0.0048635 | 0.0047467 | -2.40% | 0.93x |
| Marketing_Campaign | 0.063098 | 0.061512 | -2.51% | 1.21x |
