"""
Synthetic Dataset Generator
Student Performance Prediction Project

Each row = one student in one semester.
N_STUDENTS x N_SEMESTERS = 8,000 records.

Target:
    at_risk = 1 when final_gpa < 12
"""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_STUDENTS = 2000
N_SEMESTERS = 4

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "students.csv"


def clip(value, low, high):
    return np.clip(value, low, high)


def generate_dataset():
    rng = np.random.default_rng(SEED)

    majors = [
        "Computer Science",
        "Information Technology",
        "Management",
        "Accounting",
        "Industrial Engineering",
        "Electrical Engineering",
    ]

    records = []

    # Base characteristics for each student.
    students = {}
    for student_id in range(10001, 10001 + N_STUDENTS):
        students[student_id] = {
            "age": int(rng.integers(18, 26)),
            "gender": rng.choice(["F", "M"], p=[0.48, 0.52]),
            "major": rng.choice(majors),
            "ability": float(clip(rng.normal(0, 1), -2.5, 2.5)),
        }

    for student_id, s in students.items():

        previous_gpa = float(
            clip(14.5 + 1.2 * s["ability"] + rng.normal(0, 1.0), 8, 19.5)
        )
        cumulative_gpa = previous_gpa

        for semester in range(1, N_SEMESTERS + 1):

            age = s["age"] + (semester - 1) // 2

            failed_courses = int(
                clip(
                    rng.poisson(max(0.15, 0.8 - 0.25 * s["ability"])),
                    0,
                    6,
                )
            )

            withdrawn_courses = int(
                clip(
                    rng.poisson(max(0.10, 0.4 - 0.10 * s["ability"])),
                    0,
                    4,
                )
            )

            completed_credits = int(
                clip(
                    18 * (semester - 1)
                    + rng.normal(17, 3)
                    - failed_courses * 2,
                    0,
                    130,
                )
            )

            # Attendance
            attendance_rate = float(
                clip(
                    86
                    + 5 * s["ability"]
                    + 0.25 * (previous_gpa - 14)
                    - 3.0 * failed_courses
                    + rng.normal(0, 7),
                    50,
                    100,
                )
            )

            absence_count = int(
                clip(
                    round((100 - attendance_rate) / 3.5 + rng.normal(0, 1.3)),
                    0,
                    30,
                )
            )

            late_count = int(
                clip(
                    round(
                        2
                        - 0.6 * s["ability"]
                        + 0.10 * absence_count
                        + rng.normal(0, 1.2)
                    ),
                    0,
                    15,
                )
            )

            # Current semester performance
            midterm_average = float(
                clip(
                    0.72 * previous_gpa
                    + 1.5 * s["ability"]
                    + 0.035 * (attendance_rate - 80)
                    - 0.40 * failed_courses
                    + rng.normal(0, 1.4),
                    5,
                    20,
                )
            )

            quiz_average = float(
                clip(
                    0.78 * midterm_average
                    + 0.9 * s["ability"]
                    + rng.normal(0, 1.4),
                    5,
                    20,
                )
            )

            assignment_average = float(
                clip(
                    0.55 * previous_gpa
                    + 0.35 * midterm_average
                    + 1.0 * s["ability"]
                    + rng.normal(0, 1.3),
                    5,
                    20,
                )
            )

            project_average = float(
                clip(
                    0.40 * assignment_average
                    + 0.35 * midterm_average
                    + 0.25 * previous_gpa
                    + rng.normal(0, 1.2),
                    5,
                    20,
                )
            )

            completed_assignments = int(rng.integers(4, 11))

            late_assignments = int(
                clip(
                    round(
                        1.5
                        - 0.45 * s["ability"]
                        + 0.07 * absence_count
                        + rng.normal(0, 1.0)
                    ),
                    0,
                    completed_assignments,
                )
            )

            # LMS activity
            lms_active_days = int(
                clip(
                    round(
                        17
                        + 4.5 * s["ability"]
                        + 0.07 * (attendance_rate - 80)
                        + rng.normal(0, 3.5)
                    ),
                    3,
                    30,
                )
            )

            lms_login_count = int(
                clip(
                    round(
                        lms_active_days * 1.25
                        + 1.5 * s["ability"]
                        + rng.normal(0, 4)
                    ),
                    3,
                    60,
                )
            )

            learning_time_hours = float(
                clip(
                    1.35 * lms_active_days
                    + 0.45 * (assignment_average - 10)
                    + rng.normal(0, 6),
                    4,
                    90,
                )
            )

            # Trend features
            gpa_trend = float(
                clip(
                    previous_gpa - cumulative_gpa + rng.normal(0, 0.35),
                    -8,
                    8,
                )
            )

            current_performance = (
                0.45 * midterm_average
                + 0.20 * quiz_average
                + 0.20 * assignment_average
                + 0.15 * project_average
            )

            performance_trend = float(
                clip(
                    (current_performance - previous_gpa) / 10
                    + rng.normal(0, 0.035),
                    -1,
                    1,
                )
            )

            # Latent academic risk.
            # This controls the synthetic outcome while retaining noise.
            risk_index = (
                0.40 * (15 - previous_gpa)
                + 0.06 * (80 - attendance_rate)
                + 0.40 * (14 - midterm_average)
                + 0.50 * failed_courses
                + 0.12 * late_assignments
                - 0.05 * lms_active_days
                - 1.20 * performance_trend
                + rng.normal(0, 1.2)
            )

            # Final GPA is correlated with the above factors,
            # but is not deterministically calculated from them.
            final_gpa = float(
                clip(
                    13.7
                    - 0.43 * risk_index
                    + rng.normal(0, 1.0),
                    5,
                    20,
                )
            )

            at_risk = int(final_gpa < 12)

            records.append(
                {
                    "student_id": student_id,
                    "age": age,
                    "gender": s["gender"],
                    "major": s["major"],
                    "academic_level": "Bachelor",
                    "semester": semester,
                    "previous_gpa": round(previous_gpa, 2),
                    "cumulative_gpa": round(cumulative_gpa, 2),
                    "failed_courses": failed_courses,
                    "completed_credits": completed_credits,
                    "withdrawn_courses": withdrawn_courses,
                    "attendance_rate": round(attendance_rate, 2),
                    "absence_count": absence_count,
                    "late_count": late_count,
                    "midterm_average": round(midterm_average, 2),
                    "quiz_average": round(quiz_average, 2),
                    "assignment_average": round(assignment_average, 2),
                    "project_average": round(project_average, 2),
                    "completed_assignments": completed_assignments,
                    "late_assignments": late_assignments,
                    "lms_login_count": lms_login_count,
                    "lms_active_days": lms_active_days,
                    "learning_time_hours": round(learning_time_hours, 2),
                    "gpa_trend": round(gpa_trend, 2),
                    "performance_trend": round(performance_trend, 3),
                    "final_gpa": round(final_gpa, 2),
                    "at_risk": at_risk,
                }
            )

            cumulative_gpa = (
                cumulative_gpa * semester + final_gpa
            ) / (semester + 1)

            previous_gpa = final_gpa

    df = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset created: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["at_risk"].value_counts().sort_index())
    print("\nTarget percentage:")
    print(
        df["at_risk"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


if __name__ == "__main__":
    generate_dataset()
