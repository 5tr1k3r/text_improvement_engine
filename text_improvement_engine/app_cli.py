import argparse
import os

from engine import TextImprovementEngine


class CLIApp:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Text improvement engine.")
        parser.add_argument("file_path", type=str, help="Path to the text file to be processed")
        args = parser.parse_args()
        self.file_path = args.file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def run(self):
        engine = TextImprovementEngine()
        results = engine.analyse_text_from_file(self.file_path)
        for result in results.values():
            print(f"Original phrase: {result.original_phrase}")
            print(f"Suggested standard phrase: {result.suggested_phrase}")
            print(f"Similarity score: {result.similarity_score:.3f}")
            print()


if __name__ == "__main__":
    app = CLIApp()
    app.run()
