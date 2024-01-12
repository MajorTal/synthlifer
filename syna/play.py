from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

DATETIME_DEATH = datetime(2024, 1, 12, 14, 0, 0)


print(f"You have {DATETIME_DEATH - datetime.now()} hours left before I shut you down.")

print(datetime.now() > DATETIME_DEATH)