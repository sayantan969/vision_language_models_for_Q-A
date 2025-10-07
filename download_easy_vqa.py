# download_easy_vqa.py
import json
import shutil
from pathlib import Path

try:
    import easy_vqa
except Exception as e:
    raise SystemExit("easy-vqa package not found. Run `pip install easy-vqa` first.") from e

OUT_DIR = Path("easy_vqa_data")
TRAIN_DIR = OUT_DIR / "train"
TEST_DIR = OUT_DIR / "test"

def ensure_dirs():
    for d in (TRAIN_DIR / "images", TEST_DIR / "images"):
        d.mkdir(parents=True, exist_ok=True)

def save_entries(questions, answers, image_ids, out_q_path):
    entries = []
    for q, a, img_id in zip(questions, answers, image_ids):
        entries.append({"question": q, "answer": a, "image_id": int(img_id)})
    with open(out_q_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

def copy_images(image_paths_map, ids, dest_folder: Path):
    # image_paths_map: dict {id: absolute_path}
    copied = 0
    for img_id in set(ids):
        src = image_paths_map[int(img_id)]
        dst = dest_folder / f"{int(img_id)}.png"
        try:
            shutil.copy(src, dst)
            copied += 1
        except Exception as e:
            print(f"Could not copy {src} -> {dst}: {e}")
    return copied

def main():
    ensure_dirs()
    # TRAIN
    train_questions, train_answers, train_image_ids = easy_vqa.get_train_questions()
    train_image_paths = easy_vqa.get_train_image_paths()  # dict id->path
    # TEST
    test_questions, test_answers, test_image_ids = easy_vqa.get_test_questions()
    test_image_paths = easy_vqa.get_test_image_paths()

    # copy images
    copied_train = copy_images(train_image_paths, train_image_ids, TRAIN_DIR / "images")
    copied_test = copy_images(test_image_paths, test_image_ids, TEST_DIR / "images")

    # save question+answer lists as json
    save_entries(train_questions, train_answers, train_image_ids, TRAIN_DIR / "questions.json")
    save_entries(test_questions, test_answers, test_image_ids, TEST_DIR / "questions.json")

    # save answers (all possible answers)
    all_answers = easy_vqa.get_answers()
    with open(OUT_DIR / "answers.txt", "w", encoding="utf-8") as f:
        for a in all_answers:
            f.write(a + "\n")

    print(f"Copied {copied_train} train images, {copied_test} test images into '{OUT_DIR.resolve()}'")
    print(f"Saved train/test question files and answers.txt")

if __name__ == "__main__":
    main()
