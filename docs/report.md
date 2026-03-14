# Experiment 3 - Feature Selection Report

## Selected Features & Jaccard Similarity

| k | method1 | method1_feature | method2 | method2_feature | method3 | method3_feature | filter vs wrapper | filter vs embedded | wrapper vs embedded |
|---|---------|-----------------|---------|-----------------|---------|-----------------|-------------------|-------------------|---------------------|
| 5 | filter | "ai_generated_content_percentage", "last_exam_score", "assignment_scores_avg", "concept_understanding_score", "Copilot_usage" | wrapper | "ai_generated_content_percentage", "last_exam_score", "assignment_scores_avg", "attendance_percentage", "concept_understanding_score" | embedded | "social_media_hours", "ai_generated_content_percentage", "concept_understanding_score", "assignment_scores_avg", "last_exam_score" | 0.6666666666666666 | 0.6666666666666666 | 0.6666666666666666 |
| 7 | filter | "ai_generated_content_percentage", "last_exam_score", "assignment_scores_avg", "concept_understanding_score", "social_media_hours", "Copilot_usage", "ai_usage_purpose_Exam Prep" | wrapper | "ai_usage_time_minutes", "ai_generated_content_percentage", "last_exam_score", "assignment_scores_avg", "attendance_percentage", "concept_understanding_score", "improvement_rate" | embedded | "attendance_percentage", "improvement_rate", "social_media_hours", "ai_generated_content_percentage", "concept_understanding_score", "assignment_scores_avg", "last_exam_score" | 0.4 | 0.5555555555555556 | 0.75 |
| 10 | filter | "ai_generated_content_percentage", "ai_ethics_score", "last_exam_score", "assignment_scores_avg", "concept_understanding_score", "social_media_hours", "Copilot_usage", "Gemini_usage", "ai_usage_purpose_Doubt Solving", "ai_usage_purpose_Exam Prep" | wrapper | "ai_usage_time_minutes", "ai_generated_content_percentage", "ai_prompts_per_week", "last_exam_score", "assignment_scores_avg", "attendance_percentage", "concept_understanding_score", "study_consistency_index", "improvement_rate", "tutoring_hours" | embedded | "study_hours_per_day", "ai_prompts_per_week", "tutoring_hours", "attendance_percentage", "improvement_rate", "social_media_hours", "ai_generated_content_percentage", "concept_understanding_score", "assignment_scores_avg", "last_exam_score" | 0.25 | 0.3333333333333333 | 0.6666666666666666 |
| 15 | filter | "age", "uses_ai", "ai_generated_content_percentage", "ai_ethics_score", "last_exam_score", "assignment_scores_avg", "concept_understanding_score", "social_media_hours", "tutoring_hours", "Claude_usage", "Copilot_usage", "Gemini_usage", "gender_Male", "ai_usage_purpose_Doubt Solving", "ai_usage_purpose_Exam Prep" | wrapper | "age", "study_hours_per_day", "ai_usage_time_minutes", "ai_generated_content_percentage", "ai_prompts_per_week", "ai_ethics_score", "last_exam_score", "assignment_scores_avg", "attendance_percentage", "concept_understanding_score", "study_consistency_index", "improvement_rate", "sleep_hours", "social_media_hours", "tutoring_hours" | embedded | "age", "study_consistency_index", "ai_ethics_score", "sleep_hours", "ai_usage_time_minutes", "study_hours_per_day", "ai_prompts_per_week", "tutoring_hours", "attendance_percentage", "improvement_rate", "social_media_hours", "ai_generated_content_percentage", "concept_understanding_score", "assignment_scores_avg", "last_exam_score" | 0.36363636363636365 | 0.36363636363636365 | 1.0 |

## Metrics Comparison

| k | method | accuracy | precision | recall | f1_score | max_depth | n_leaves |
|---|--------|----------|-----------|--------|----------|-----------|----------|
| 5 | filter | 0.913125 | 0.9508 | 0.9515 | 0.9512 | 23 | 486 |
| 5 | wrapper | 0.920625 | 0.9570 | 0.9536 | 0.9553 | 21 | 437 |
| 5 | embedded | 0.915000 | 0.9529 | 0.9515 | 0.9522 | 21 | 460 |
| 7 | filter | 0.914375 | 0.9497 | 0.9543 | 0.9520 | 21 | 445 |
| 7 | wrapper | 0.913750 | 0.9515 | 0.9515 | 0.9515 | 18 | 401 |
| 7 | embedded | 0.905000 | 0.9466 | 0.9466 | 0.9466 | 20 | 394 |
| 10 | filter | 0.909375 | 0.9487 | 0.9494 | 0.9491 | 20 | 433 |
| 10 | wrapper | 0.916250 | 0.9542 | 0.9515 | 0.9529 | 18 | 351 |
| 10 | embedded | 0.913750 | 0.9509 | 0.9522 | 0.9515 | 20 | 362 |
| 15 | filter | 0.908750 | 0.9487 | 0.9487 | 0.9487 | 21 | 398 |
| 15 | wrapper | 0.909375 | 0.9494 | 0.9487 | 0.9490 | 19 | 332 |
| 15 | embedded | 0.907500 | 0.9467 | 0.9494 | 0.9481 | 20 | 338 |

## Feature Importance

### k = 5

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4443 | 0.4247 | 0.4311 |
| assignment_scores_avg | 0.2652 | 0.2466 | 0.2447 |
| concept_understanding_score | 0.1743 | 0.1704 | 0.1724 |
| ai_generated_content_percentage | 0.1087 | 0.0860 | 0.0842 |
| Copilot_usage | 0.0075 | - | - |
| attendance_percentage | - | 0.0723 | - |
| social_media_hours | - | - | 0.0677 |

### k = 7

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4248 | 0.4070 | 0.4122 |
| assignment_scores_avg | 0.2396 | 0.2250 | 0.2156 |
| concept_understanding_score | 0.1692 | 0.1591 | 0.1601 |
| ai_generated_content_percentage | 0.0822 | 0.0685 | 0.0583 |
| social_media_hours | 0.0655 | - | 0.0545 |
| ai_usage_purpose_Exam Prep | 0.0106 | - | - |
| Copilot_usage | 0.0081 | - | - |
| improvement_rate | - | 0.0609 | 0.0519 |
| ai_usage_time_minutes | - | 0.0451 | - |
| attendance_percentage | - | 0.0343 | 0.0473 |

### k = 10

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4159 | 0.3888 | 0.3961 |
| assignment_scores_avg | 0.2262 | 0.1965 | 0.2041 |
| concept_understanding_score | 0.1673 | 0.1583 | 0.1535 |
| ai_generated_content_percentage | 0.0734 | 0.0498 | 0.0417 |
| social_media_hours | 0.0468 | - | 0.0314 |
| ai_ethics_score | 0.0324 | - | - |
| Gemini_usage | 0.0138 | - | - |
| ai_usage_purpose_Exam Prep | 0.0095 | - | - |
| ai_usage_purpose_Doubt Solving | 0.0094 | - | - |
| Copilot_usage | 0.0054 | - | - |
| ai_prompts_per_week | - | 0.0438 | 0.0475 |
| improvement_rate | - | 0.0390 | 0.0413 |
| ai_usage_time_minutes | - | 0.0351 | - |
| attendance_percentage | - | 0.0314 | 0.0343 |
| study_consistency_index | - | 0.0305 | - |
| tutoring_hours | - | 0.0267 | 0.0259 |
| study_hours_per_day | - | - | 0.0245 |

### k = 15

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4113 | 0.3855 | 0.3842 |
| assignment_scores_avg | 0.2133 | 0.1938 | 0.1979 |
| concept_understanding_score | 0.1626 | 0.1543 | 0.1548 |
| ai_generated_content_percentage | 0.0510 | 0.0313 | 0.0315 |
| tutoring_hours | 0.0441 | 0.0230 | 0.0224 |
| social_media_hours | 0.0328 | 0.0250 | 0.0256 |
| age | 0.0248 | 0.0107 | 0.0120 |
| ai_ethics_score | 0.0218 | 0.0122 | 0.0136 |
| Gemini_usage | 0.0110 | - | - |
| uses_ai | 0.0068 | - | - |
| ai_usage_purpose_Exam Prep | 0.0058 | - | - |
| Claude_usage | 0.0058 | - | - |
| ai_usage_purpose_Doubt Solving | 0.0036 | - | - |
| gender_Male | 0.0035 | - | - |
| Copilot_usage | 0.0018 | - | - |
| improvement_rate | - | 0.0395 | 0.0309 |
| ai_prompts_per_week | - | 0.0335 | 0.0341 |
| ai_usage_time_minutes | - | 0.0224 | 0.0237 |
| study_hours_per_day | - | 0.0222 | 0.0248 |
| study_consistency_index | - | 0.0205 | 0.0211 |
| sleep_hours | - | 0.0142 | 0.0151 |
| attendance_percentage | - | 0.0120 | 0.0083 |
