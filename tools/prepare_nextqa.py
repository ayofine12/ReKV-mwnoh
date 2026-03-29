import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_video_map(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_split(csv_path, video_root, video_map):
    grouped_samples = defaultdict(list)
    video_meta = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video"]
            mapped_video = video_map.get(video_id)
            if mapped_video is None:
                raise KeyError(f"Missing video mapping for {video_id}")

            video_path = Path(video_root) / f"{mapped_video}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            answer_idx = int(row["answer"])
            choices = [row[f"a{i}"] for i in range(5)]

            grouped_samples[video_id].append(
                {
                    "question_idx": row["qid"],
                    "question": row["question"],
                    "choices": choices,
                    "answer": choices[answer_idx],
                    "question_type": row["type"],
                }
            )

            if video_id not in video_meta:
                video_meta[video_id] = {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "frame_count": int(row["frame_count"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "dataset": "nextqa",
                }

    converted = []
    for video_id, conversations in grouped_samples.items():
        sample = dict(video_meta[video_id])
        sample["conversations"] = conversations
        converted.append(sample)

    converted.sort(key=lambda item: item["video_id"])
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nextqa-root",
        type=Path,
        default=Path("/mnt/ssd1/mwnoh/NExT-QA"),
        help="Path to the NExT-QA repository root.",
    )
    parser.add_argument(
        "--nextvideo-root",
        type=Path,
        default=Path("/mnt/ssd1/mwnoh/NExTVideo"),
        help="Path to the NExTVideo directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/nextqa"),
        help="Directory where converted ReKV annotations will be written.",
    )
    args = parser.parse_args()

    dataset_dir = args.nextqa_root / "dataset" / "nextqa"
    map_path = dataset_dir / "map_vid_vidorID.json"
    video_map = load_video_map(map_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        csv_path = dataset_dir / f"{split}.csv"
        converted = convert_split(csv_path, args.nextvideo_root, video_map)
        output_path = args.output_dir / f"{split}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(converted)} videos to {output_path}")


if __name__ == "__main__":
    main()
