from pathlib import Path

def main():
    cwd = Path.cwd()
    pipelines_dir = cwd / ".pipeline"

    if not pipelines_dir.is_dir():
        print("error: .pipelines directory not found")
        return

    # collect prefixes from .p-lock and .p-log files
    pipeline_files = list(pipelines_dir.glob("*.p-lock")) + list(pipelines_dir.glob("*.p-log"))

    for pfile in pipeline_files:
        prefix = pfile.name.split(".", 1)[0]

        # check for any file in cwd with the same prefix
        has_match = any(
            f.is_file() and f.name.startswith(prefix + ".")
            for f in cwd.iterdir()
        )

        if not has_match:
            pfile.unlink()

if __name__ == "__main__":
    main()

