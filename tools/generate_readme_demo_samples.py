from __future__ import annotations

from tools.generate_audio_samples import ROOT, generate_readme_demo_samples


def main() -> None:
    for output in generate_readme_demo_samples():
        print(f"generated {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
