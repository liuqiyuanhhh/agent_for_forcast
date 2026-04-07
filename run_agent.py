import json
import sys

from src.agent import predict


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "events.sample.json"
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)

    if isinstance(events, dict):
        events = [events]

    for i, event in enumerate(events, start=1):
        print(f"[{i}] {event.get('title', 'N/A')}")
        print(json.dumps(predict(event), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
