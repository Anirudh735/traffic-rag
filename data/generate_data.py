# faker is a library that generates realistic fake data
# We use it to simulate 250 traffic violation records
from faker import Faker
import json
import random
from datetime import datetime, timedelta

# Create a Faker instance — this is our fake data generator
fake = Faker("en_IN")  # en_IN = Indian English locale, so names/places feel local

# --- Define the building blocks of a violation record ---

# These are the cameras placed at real Bangalore intersections
CAMERAS = [
    "CAM-001 MG Road",
    "CAM-002 Silk Board",
    "CAM-003 Hebbal Flyover",
    "CAM-004 Marathahalli Bridge",
    "CAM-005 Koramangala 5th Block",
    "CAM-006 Whitefield Main Road",
    "CAM-007 Bannerghatta Road",
    "CAM-008 Yeshwanthpur Circle",
]

# These are the types of violations our YOLO system can detect
VIOLATION_TYPES = [
    "red light jumping",
    "speeding",
    "wrong lane driving",
    "no helmet",
    "triple riding",
    "signal violation",
    "illegal parking",
]

# Karnataka vehicle plates follow the format KA-XX-XX-XXXX
def generate_plate():
    district = random.randint(1, 99)       # district number
    series = random.choice("ABCDEFGHKLMNPRSTUVY")  # letter series
    number = random.randint(1000, 9999)    # 4-digit number
    return f"KA-{district:02d}-{series}-{number}"


# This function creates ONE violation record as a Python dictionary
def generate_violation():
    # Generate a random timestamp within the last 30 days
    days_ago = random.randint(0, 30)
    hour = random.randint(6, 22)           # violations happen 6am to 10pm
    minute = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago)
    timestamp = timestamp.replace(hour=hour, minute=minute, second=0)

    return {
        "plate": generate_plate(),
        "camera": random.choice(CAMERAS),
        "violation": random.choice(VIOLATION_TYPES),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "speed_kmph": random.randint(45, 110) if "speeding" in VIOLATION_TYPES else None,
        "fine_amount": random.choice([500, 1000, 1500, 2000]),
        "status": random.choice(["pending", "paid", "contested"]),
    }


# --- Generate 250 records and save them ---

# A list comprehension: runs generate_violation() 250 times
# and collects all results into a list
violations = [generate_violation() for _ in range(250)]

# Save to a JSON file so we can load it later during ingestion
with open("data/violations.json", "w") as f:
    json.dump(violations, f, indent=2)

print(f"Generated {len(violations)} violation records")
print(f"Sample record:\n{json.dumps(violations[0], indent=2)}")