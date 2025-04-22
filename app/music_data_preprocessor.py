import pickle as pkl

from pathlib import Path

from .music_converter import MusicConverter

BASE_DIR = Path(__file__).parent

class MusicDataPreprocessor:
    """Handles music dataset preprocessing including format conversion and file management.

    Manages conversion between various music formats (MXL, MusicXML, **kern) to LMX format,
    with directory structure preservation.
    """
    def __init__(self) -> None:
        """Initialize paths for raw and processed datasets."""
        self.data_dir = BASE_DIR / "datasets/"
        self.converted_lmx_dir = BASE_DIR / "converted_dataset/"
        self.lmx_files: list[str]

        if not self.data_dir.exists():
            self.data_dir.mkdir()
        if not self.converted_lmx_dir.exists():
            self.converted_lmx_dir.mkdir()

    def save_to_pickle(self, data: list[str], filename: str) -> None:
        """Serialize LMX data to binary pickle format.

        Args:
            data (list[str]): LMX sequences to save
            filename (str): Target filename in converted_dataset directory
        """
        with open(self.converted_lmx_dir / filename, "wb") as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

    def load_from_pickle(self, filename: str) -> list[str]:
        """Load serialized LMX data from pickle file.

        Args:
            filename (str): Source filename in converted_dataset directory

        Returns:
            list[str]: Deserialized LMX sequences
        """
        path = self.converted_lmx_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        with open(path, "rb") as f:
            data = pkl.load(f)
        return data

    def load_all_lmx_files(
        self, only_subdir: str | None = None, allowed_files: set[str] | None = None
    ) -> list[str]:
        """Load LMX files with optional directory and filename filtering.

        Args:
            only_subdir (str, optional): Restrict to specific subdirectory
            allowed_files (set[str], optional): Filter by specific filenames

        Returns:
            list[str]: Loaded LMX content strings
        """
        lmx_files = []
        directory = (
            self.converted_lmx_dir / only_subdir
            if only_subdir is not None
            else self.converted_lmx_dir
        )
        for root, dirs, files in directory.walk():
            for file in files:
                if file not in allowed_files:
                    continue
                filepath = root / file
                try:
                    if not filepath.suffix == ".lmx":
                        continue
                    with open(filepath, "r") as f:
                        lmx_file = f.read()
                        lmx_files.append(lmx_file)
                except Exception as e:
                    print(e)
        return lmx_files

    def convert_data_to_lmx(self, allowed_files: set[str]) -> None:
        """Convert supported music formats to LMX with directory structure preservation.

        Args:
            allowed_files (set[str]): Whitelist of filenames to process

        Converts:
            - .mxl → LMX
            - .musicxml → LMX
            - .krn (**kern) → LMX
        """
        for root, dirs, files in self.data_dir.walk():
            dir_created = False
            for file in files:
                if file not in allowed_files:
                    continue
                filepath = root / file
                suffix = filepath.suffix
                lmx_string: str
                new_lmx_dirpath = Path(
                    self.converted_lmx_dir, root.relative_to(self.data_dir)
                )
                new_lmx_filename = new_lmx_dirpath / Path(file).with_suffix(".lmx")
                if new_lmx_filename.exists():
                    continue
                try:
                    match suffix:
                        case ".lmx":
                            continue
                        case ".mxl":
                            lmx_string = MusicConverter.mxl_to_lmx(filepath)
                        case ".musicxml":
                            lmx_string = MusicConverter.musicxml_to_lmx(filepath)
                        case ".krn":
                            lmx_string = MusicConverter.kern_to_lmx(filepath)
                        case _:
                            continue
                except Exception as e:
                    print(f"Converting of file {filepath} failed. {e}.")
                    continue
                if lmx_string == "":
                    print(f"Converting of file {filepath} failed.")
                    continue
                if not dir_created:
                    # use local variable to prevent repetitive io checks of dir existence
                    new_lmx_dirpath.mkdir(parents=True, exist_ok=True)
                    dir_created = True
                with open(new_lmx_filename, "w") as lmx_file:
                    lmx_file.write(lmx_string)
