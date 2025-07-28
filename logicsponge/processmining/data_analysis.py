from pathlib import Path

import pandas as pd

# Load your CSV file
# Build the path as an absolute path from the directory of this file
file_path = Path(__file__).resolve().parent / "../../data/Sepsis_Cases.csv"
df = pd.read_csv(file_path)

# Ensure the timestamp column is in datetime format
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

# Number of Events per Case
case_sizes = df["case:concept:name"].value_counts()
case_sizes_description = case_sizes.describe()
print("Case Sizes Description:\n", case_sizes_description)

# Duration of Each Case
case_durations = df.groupby("case:concept:name")["time:timestamp"].agg(lambda x: x.max() - x.min())
case_durations_description = case_durations.describe()
# print("Case Durations Description:\n", case_durations_description)

# Count of Each Phase per Case
phase_counts = df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
phase_counts_description = phase_counts.describe()
print("Phase Counts Description:\n", phase_counts_description)


# Consecutive Phase Repetitions in Cases
def max_consecutive_repeats(phase_list):
    max_repeats = {}
    current_activity = phase_list[0]
    current_count = 1

    for i in range(1, len(phase_list)):
        if phase_list[i] == current_activity:
            current_count += 1
        else:
            if current_activity in max_repeats:
                max_repeats[current_activity] = max(max_repeats[current_activity], current_count)
            else:
                max_repeats[current_activity] = current_count
            current_activity = phase_list[i]
            current_count = 1

    if current_activity in max_repeats:
        max_repeats[current_activity] = max(max_repeats[current_activity], current_count)
    else:
        max_repeats[current_activity] = current_count

    return max_repeats


# Apply the function to each case
continuous_repeats = df.groupby("case:concept:name")["concept:name"].agg(lambda x: max_consecutive_repeats(x.tolist()))

# Flatten the dictionary to create a DataFrame
continuous_repeats_df = continuous_repeats.apply(pd.Series).fillna(0)
continuous_repeats_description = continuous_repeats_df.describe()
print("Continuous Repeats Description:\n", continuous_repeats_description)

summary_df = pd.DataFrame(
    {
        "num_events": case_sizes,
        "case_duration": case_durations.dt.total_seconds(),
        "max_continuous_phase": continuous_repeats_df.max(axis=1),
    }
)
summary_df_description = summary_df.describe()
print("Summary DataFrame Description:\n", summary_df_description)

# Save descriptions to a CSV file
output_path = Path(__file__).resolve().parent / "../../data/description_output.csv"
with open(output_path, "w") as f:
    f.write("Case Sizes Description:\n")
    case_sizes_description.to_csv(f)
    f.write("\nCase Durations Description:\n")
    case_durations_description.to_csv(f)
    f.write("\nPhase Counts Description:\n")
    phase_counts_description.to_csv(f)
    f.write("\nContinuous Repeats Description:\n")
    continuous_repeats_description.to_csv(f)
    f.write("\nSummary DataFrame Description:\n")
    summary_df_description.to_csv(f)
