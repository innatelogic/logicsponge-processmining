import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
MAX_NUM_CASES = 50
TOTAL_ACTIVITIES = 50000
start_time = datetime(2013, 11, 7, 8, 0, 0)
time_increment = timedelta(minutes=5)

# Track case states
case_ids = []
case_counters = {}
next_case_num = 1
current_time = start_time

# Generate CSV (always save into the repository's data/ folder, regardless of CWD)
# The examples/ file lives in <repo>/examples, so go up one level and into data/
data_dir = Path(__file__).resolve().parents[1] / "data"
data_dir.mkdir(parents=True, exist_ok=True)
save_path = data_dir / "Synthetic111000.csv"

with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)

    # Write header with only the three required columns
    writer.writerow(["case:concept:name", "concept:name", "time:timestamp"])

    for _ in range(TOTAL_ACTIVITIES):
        num_cases = len(case_ids)

        # Decide whether to pick existing or create new case
        rand_val = random.random()

        if rand_val < num_cases / MAX_NUM_CASES and num_cases > 0:
            # Pick random existing case
            case_id = random.choice(case_ids)
        else:
            # Create new case
            case_id = f"case_{next_case_num}"
            case_ids.append(case_id)
            case_counters[case_id] = 0
            next_case_num += 1

        # Determine activity based on pattern: 3x act_0, then 3x act_1, repeat
        event_num = case_counters[case_id]
        cycle_position = event_num % 6
        activity = "act_0" if cycle_position < 3 else "act_1"

        # Format timestamp
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S+00:00")

        # Write row
        writer.writerow([case_id, activity, timestamp])

        # Update state
        case_counters[case_id] += 1
        current_time += time_increment

print(f"CSV file generated successfully at: {save_path}")
print("Total activities: {}".format(TOTAL_ACTIVITIES))
print("Total unique cases: {}".format(len(case_ids)))