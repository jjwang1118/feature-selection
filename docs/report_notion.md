# Experiment 3 - Feature Selection Report

---

## Selected Features & Jaccard Similarity

### k = 5

| method | selected features |
|--------|-------------------|
| filter | ai_generated_content_percentage, last_exam_score, assignment_scores_avg, concept_understanding_score, Copilot_usage |
| wrapper | ai_generated_content_percentage, ai_prompts_per_week, last_exam_score, assignment_scores_avg, concept_understanding_score |
| embedded | ai_generated_content_percentage, ai_prompts_per_week, concept_understanding_score, assignment_scores_avg, last_exam_score |

| comparison | jaccard similarity |
|------------|-------------------|
| filter vs wrapper | 0.6666666666666666 |
| filter vs embedded | 0.6666666666666666 |
| wrapper vs embedded | 1.0 |

### k = 7

| method | selected features |
|--------|-------------------|
| filter | ai_generated_content_percentage, last_exam_score, assignment_scores_avg, concept_understanding_score, social_media_hours, Copilot_usage, ai_usage_purpose_Exam Prep |
| wrapper | ai_generated_content_percentage, ai_prompts_per_week, last_exam_score, assignment_scores_avg, concept_understanding_score, improvement_rate, social_media_hours |
| embedded | tutoring_hours, improvement_rate, ai_generated_content_percentage, ai_prompts_per_week, concept_understanding_score, assignment_scores_avg, last_exam_score |

| comparison | jaccard similarity |
|------------|-------------------|
| filter vs wrapper | 0.5555555555555556 |
| filter vs embedded | 0.4 |
| wrapper vs embedded | 0.75 |

### k = 10

| method | selected features |
|--------|-------------------|
| filter | ai_generated_content_percentage, ai_ethics_score, last_exam_score, assignment_scores_avg, concept_understanding_score, social_media_hours, Copilot_usage, Gemini_usage, ai_usage_purpose_Doubt Solving, ai_usage_purpose_Exam Prep |
| wrapper | study_hours_per_day, ai_usage_time_minutes, ai_generated_content_percentage, ai_prompts_per_week, last_exam_score, assignment_scores_avg, concept_understanding_score, study_consistency_index, improvement_rate, social_media_hours |
| embedded | study_consistency_index, social_media_hours, study_hours_per_day, tutoring_hours, improvement_rate, ai_generated_content_percentage, ai_prompts_per_week, concept_understanding_score, assignment_scores_avg, last_exam_score |

| comparison | jaccard similarity |
|------------|-------------------|
| filter vs wrapper | 0.3333333333333333 |
| filter vs embedded | 0.3333333333333333 |
| wrapper vs embedded | 0.8181818181818182 |

### k = 15

| method | selected features |
|--------|-------------------|
| filter | age, uses_ai, ai_generated_content_percentage, ai_ethics_score, last_exam_score, assignment_scores_avg, concept_understanding_score, social_media_hours, tutoring_hours, Claude_usage, Copilot_usage, Gemini_usage, gender_Male, ai_usage_purpose_Doubt Solving, ai_usage_purpose_Exam Prep |
| wrapper | study_hours_per_day, ai_usage_time_minutes, ai_dependency_score, ai_generated_content_percentage, ai_prompts_per_week, last_exam_score, assignment_scores_avg, attendance_percentage, concept_understanding_score, study_consistency_index, improvement_rate, sleep_hours, social_media_hours, tutoring_hours, class_participation_score |
| embedded | grade_level, ai_ethics_score, ai_usage_time_minutes, sleep_hours, ai_dependency_score, study_consistency_index, social_media_hours, study_hours_per_day, tutoring_hours, improvement_rate, ai_generated_content_percentage, ai_prompts_per_week, concept_understanding_score, assignment_scores_avg, last_exam_score |

| comparison | jaccard similarity |
|------------|-------------------|
| filter vs wrapper | 0.25 |
| filter vs embedded | 0.30434782608695654 |
| wrapper vs embedded | 0.7647058823529411 |

---

## Metrics Comparison

### k = 5

| method | accuracy | precision | recall | f1_score | max_depth | n_leaves |
|--------|----------|-----------|--------|----------|-----------|----------|
| filter | 0.913125 | 0.9508 | 0.9515 | 0.9512 | 23 | 486 |
| wrapper | 0.912500 | 0.9534 | 0.9480 | 0.9507 | 20 | 450 |
| embedded | 0.913750 | 0.9541 | 0.9487 | 0.9514 | 20 | 448 |

### k = 7

| method | accuracy | precision | recall | f1_score | max_depth | n_leaves |
|--------|----------|-----------|--------|----------|-----------|----------|
| filter | 0.914375 | 0.9497 | 0.9543 | 0.9520 | 21 | 445 |
| wrapper | 0.906250 | 0.9473 | 0.9473 | 0.9473 | 21 | 388 |
| embedded | 0.921250 | 0.9583 | 0.9529 | 0.9556 | 20 | 396 |

### k = 10

| method | accuracy | precision | recall | f1_score | max_depth | n_leaves |
|--------|----------|-----------|--------|----------|-----------|----------|
| filter | 0.909375 | 0.9487 | 0.9494 | 0.9491 | 20 | 433 |
| wrapper | 0.905625 | 0.9479 | 0.9459 | 0.9469 | 20 | 349 |
| embedded | 0.913750 | 0.9484 | 0.9550 | 0.9517 | 19 | 356 |

### k = 15

| method | accuracy | precision | recall | f1_score | max_depth | n_leaves |
|--------|----------|-----------|--------|----------|-----------|----------|
| filter | 0.908750 | 0.9487 | 0.9487 | 0.9487 | 21 | 398 |
| wrapper | 0.912500 | 0.9464 | 0.9557 | 0.9510 | 18 | 334 |
| embedded | 0.913125 | 0.9490 | 0.9536 | 0.9513 | 19 | 337 |

---

## Feature Importance

### k = 5

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4443 | 0.4276 | 0.4279 |
| assignment_scores_avg | 0.2652 | 0.2471 | 0.2452 |
| concept_understanding_score | 0.1743 | 0.1708 | 0.1697 |
| ai_generated_content_percentage | 0.1087 | 0.0876 | 0.0882 |
| Copilot_usage | 0.0075 | - | - |
| ai_prompts_per_week | - | 0.0670 | 0.0690 |

### k = 7

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4248 | 0.4098 | 0.4101 |
| assignment_scores_avg | 0.2396 | 0.2193 | 0.2116 |
| concept_understanding_score | 0.1692 | 0.1558 | 0.1666 |
| ai_generated_content_percentage | 0.0822 | 0.0570 | 0.0594 |
| social_media_hours | 0.0655 | 0.0533 | - |
| ai_usage_purpose_Exam Prep | 0.0106 | - | - |
| Copilot_usage | 0.0081 | - | - |
| ai_prompts_per_week | - | 0.0568 | 0.0539 |
| improvement_rate | - | 0.0481 | 0.0582 |
| tutoring_hours | - | - | 0.0401 |

### k = 10

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4159 | 0.3893 | 0.3948 |
| assignment_scores_avg | 0.2262 | 0.2136 | 0.2052 |
| concept_understanding_score | 0.1673 | 0.1571 | 0.1547 |
| ai_generated_content_percentage | 0.0734 | 0.0410 | 0.0385 |
| social_media_hours | 0.0468 | 0.0331 | 0.0336 |
| ai_ethics_score | 0.0324 | - | - |
| Gemini_usage | 0.0138 | - | - |
| ai_usage_purpose_Exam Prep | 0.0095 | - | - |
| ai_usage_purpose_Doubt Solving | 0.0094 | - | - |
| Copilot_usage | 0.0054 | - | - |
| ai_prompts_per_week | - | 0.0372 | 0.0402 |
| improvement_rate | - | 0.0363 | 0.0328 |
| ai_usage_time_minutes | - | 0.0340 | - |
| study_hours_per_day | - | 0.0322 | 0.0366 |
| study_consistency_index | - | 0.0260 | 0.0361 |
| tutoring_hours | - | - | 0.0275 |

### k = 15

| feature | filter | wrapper | embedded |
|---------|--------|---------|----------|
| last_exam_score | 0.4113 | 0.3832 | 0.3842 |
| assignment_scores_avg | 0.2133 | 0.1985 | 0.1960 |
| concept_understanding_score | 0.1626 | 0.1533 | 0.1542 |
| ai_generated_content_percentage | 0.0510 | 0.0308 | 0.0329 |
| tutoring_hours | 0.0441 | 0.0206 | 0.0227 |
| social_media_hours | 0.0328 | 0.0257 | 0.0293 |
| age | 0.0248 | - | - |
| ai_ethics_score | 0.0218 | - | 0.0096 |
| Gemini_usage | 0.0110 | - | - |
| uses_ai | 0.0068 | - | - |
| ai_usage_purpose_Exam Prep | 0.0058 | - | - |
| Claude_usage | 0.0058 | - | - |
| ai_usage_purpose_Doubt Solving | 0.0036 | - | - |
| gender_Male | 0.0035 | - | - |
| Copilot_usage | 0.0018 | - | - |
| ai_prompts_per_week | - | 0.0314 | 0.0351 |
| study_consistency_index | - | 0.0267 | 0.0204 |
| study_hours_per_day | - | 0.0264 | 0.0239 |
| improvement_rate | - | 0.0243 | 0.0300 |
| sleep_hours | - | 0.0216 | 0.0139 |
| ai_usage_time_minutes | - | 0.0201 | 0.0227 |
| ai_dependency_score | - | 0.0138 | 0.0180 |
| attendance_percentage | - | 0.0119 | - |
| class_participation_score | - | 0.0118 | - |
| grade_level | - | - | 0.0072 |
