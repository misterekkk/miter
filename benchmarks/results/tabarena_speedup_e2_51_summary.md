# TabArena lite, 51 tasks: E2 {'oblique_sample_rows': 4096} vs MITER_DEFAULT_V1

- tasks: 51; wins 8, ties 28, losses 15
- relative error: median +0.000%, mean +0.283%, worst +8.87%
- one-sided Wilcoxon (variant worse): p = 0.1056
- train time (8-fold bag, this machine): total 0.80x, median per task 0.90x

| task | default err | variant err | rel | time ratio |
|---|---:|---:|---:|---:|
| APSFailure | 0.0048635 | 0.0052949 | +8.87% | 0.62x |
| SDSS17 | 0.077849 | 0.080443 | +3.33% | 0.74x |
| physiochemical_protein | 3.4251 | 3.4948 | +2.03% | 0.78x |
| NATICUSdroid | 0.014063 | 0.014347 | +2.02% | 0.86x |
| houses | 0.21088 | 0.21323 | +1.11% | 0.66x |
| jm1 | 0.24067 | 0.24305 | +0.99% | 0.79x |
| credit_card_clients_default | 0.20753 | 0.20903 | +0.72% | 0.66x |
| coil2000_insurance_policies | 0.24484 | 0.24651 | +0.68% | 0.60x |
| heloc | 0.20255 | 0.20326 | +0.35% | 1.07x |
| miami_housing | 82957 | 83178 | +0.27% | 0.89x |
| kddcup09_appetency | 0.18021 | 0.18064 | +0.24% | 0.74x |
| diamonds | 531.22 | 532.4 | +0.22% | 0.86x |
| superconductivity | 9.3059 | 9.3264 | +0.22% | 0.95x |
| GiveMeSomeCredit | 0.13099 | 0.1311 | +0.09% | 0.60x |
| Diabetes130US | 0.32691 | 0.32717 | +0.08% | 0.85x |
| diabetes | 0.16908 | 0.16908 | +0.00% | 0.98x |
| maternal_health_risk | 0.43515 | 0.43515 | +0.00% | 0.99x |
| students_dropout_and_academic_success | 0.55455 | 0.55455 | +0.00% | 1.00x |
| seismic-bumps | 0.19468 | 0.19468 | +0.00% | 0.86x |
| taiwanese_bankruptcy_prediction | 0.062777 | 0.062777 | +0.00% | 0.74x |
| website_phishing | 0.22582 | 0.22582 | +0.00% | 1.01x |
| hiva_agnostic | 0.17392 | 0.17392 | +0.00% | 0.90x |
| polish_companies_bankruptcy | 0.032599 | 0.032599 | +0.00% | 1.01x |
| healthcare_insurance_expenses | 4105 | 4105 | +0.00% | 0.98x |
| qsar-biodeg | 0.074584 | 0.074584 | +0.00% | 0.99x |
| hazelnut-spread-contaminant-detection | 0.026737 | 0.026737 | +0.00% | 1.04x |
| splice | 0.091402 | 0.091402 | +0.00% | 0.81x |
| credit-g | 0.2438 | 0.2438 | +0.00% | 0.97x |
| Marketing_Campaign | 0.063098 | 0.063098 | +0.00% | 0.79x |
| QSAR_fish_toxicity | 0.89217 | 0.89217 | +0.00% | 0.96x |
| Another-Dataset-on-used-Fiat-500 | 690.76 | 690.76 | +0.00% | 0.90x |
| Bioresponse | 0.111 | 0.111 | +0.00% | 0.88x |
| Fitness_Club | 0.19606 | 0.19606 | +0.00% | 1.19x |
| Is-this-a-good-customer | 0.29163 | 0.29163 | +0.00% | 0.98x |
| MIC | 0.4987 | 0.4987 | +0.00% | 0.96x |
| Amazon_employee_access | 0.21391 | 0.21391 | +0.00% | 0.84x |
| QSAR-TID-11 | 0.84496 | 0.84496 | +0.00% | 0.80x |
| wine_quality | 0.60842 | 0.60842 | +0.00% | 1.06x |
| airfoil_self_noise | 1.2126 | 1.2126 | +0.00% | 0.94x |
| anneal | 0.058522 | 0.058522 | +0.00% | 0.99x |
| blood-transfusion-service-center | 0.29754 | 0.29754 | +0.00% | 1.49x |
| churn | 0.068081 | 0.068081 | +0.00% | 0.76x |
| concrete_compressive_strength | 4.4039 | 4.4039 | +0.00% | 1.01x |
| Food_Delivery_Time | 7.4421 | 7.4343 | -0.10% | 0.97x |
| bank-marketing | 0.22495 | 0.2247 | -0.11% | 0.93x |
| online_shoppers_intention | 0.07309 | 0.072874 | -0.30% | 1.00x |
| E-CommereShippingData | 0.2589 | 0.25812 | -0.30% | 1.02x |
| HR_Analytics_Job_Change_of_Data_Scientists | 0.19124 | 0.19041 | -0.44% | 0.60x |
| Bank_Customer_Churn | 0.1258 | 0.12432 | -1.18% | 0.79x |
| in_vehicle_coupon_recommendation | 0.17488 | 0.17126 | -2.07% | 0.77x |
| customer_satisfaction_in_airline | 0.0058347 | 0.0057005 | -2.30% | 0.79x |
