import datetime
import xml.etree.ElementTree as ET
import sys
import music21
import yaml

from pathlib import Path

from .olimpic_icdar24.app.linearization.Linearizer import Linearizer
from .olimpic_icdar24.app.linearization.Delinearizer import Delinearizer
from .olimpic_icdar24.app.symbolic.MxlFile import MxlFile
from .olimpic_icdar24.app.symbolic.part_to_score import part_to_score

BASE_DIR = Path(__file__).parent

try:
    with open(BASE_DIR / "config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    music21.environment.set("musicxmlPath", config.get("MUSESCORE_INSTALL_FILEPATH"))
    print("MuseScore installation path set successfully")
except Exception as e:
    print(
        "MuseScore installation was not set properly! Check if MUSESCORE_INSTALL_FILEPATH in config.yaml is set"
    )


class MusicConverter:
    """Handles conversion between various music notation formats including LMX, MusicXML, and **kern.

    Provides bidirectional conversion capabilities and MuseScore normalization features.
    """

    @staticmethod
    def normalize_musicxml(input_file: str) -> str | None:
        """Normalize MusicXML using MuseScore's formatting standards.

        Args:
            input_file (str): Path to MusicXML file or raw MusicXML string

        Returns:
            str | None: Normalized MusicXML content or None if normalization fails
        """
        try:  # Load using Music21
            score = music21.converter.parse(input_file)
            normalized_xml_path = score.write(
                fmt="musicxml"
            )  # Saves into temporary file
        except Exception as e:
            print("MusicXML export exception", e)
            return None
        with open(normalized_xml_path) as f:
            normalized_xml = f.read()
        return normalized_xml

    @staticmethod
    def kern_to_lmx(data: Path | str) -> str:
        """Convert **kern format to LMX tokens.

        Args:
            data (Path | str): **kern file path or raw data

        Returns:
            str: LMX token sequence

        Raises:
            TypeError: For unsupported input types
        """
        humdrum_converter = music21.converter.subConverters.ConverterHumdrum()
        if isinstance(data, Path):
            score = humdrum_converter.parseFile(data)
        elif isinstance(data, str):
            score = humdrum_converter.parseData(data)
        else:
            raise TypeError("Unsupported data type")
        musicxml = music21.converter.toData(score.stream, "musicxml")
        musicxml = MxlFile(ET.ElementTree(ET.fromstring(musicxml)))
        return MusicConverter._musicxml_file_to_lmx(musicxml)

    @staticmethod
    def mxl_to_lmx(filepath: Path) -> str:
        """Convert compressed MusicXML (.mxl) to LMX tokens.

        Args:
            filepath (Path): Path to .mxl file

        Returns:
            str: LMX token sequence
        """
        musicxml = MxlFile.load_mxl(str(filepath))
        return MusicConverter._musicxml_file_to_lmx(musicxml)

    @staticmethod
    def musicxml_to_lmx(data: Path | str) -> str:
        """Convert MusicXML to LMX tokens.

        Args:
            data (Path | str): MusicXML file path or raw data

        Returns:
            str: LMX token sequence
        """
        xml = MusicConverter._get_data(data)
        musicxml = MxlFile(ET.ElementTree(ET.fromstring(xml)))
        return MusicConverter._musicxml_file_to_lmx(musicxml)

    @staticmethod
    def _musicxml_file_to_lmx(mxl: MxlFile):
        """Internal method to convert MxlFile object to LMX tokens.

        Args:
            mxl (MxlFile): Parsed MusicXML file object

        Returns:
            str: LMX token sequence

        Note:
            Exits program if no valid <part> element is found
        """
        try:
            part = mxl.get_piano_part()
        except:
            part = mxl.tree.find("part")

        if part is None or part.tag != "part":
            print("No <part> element found.", file=sys.stderr)
            exit()

        linearizer = Linearizer(errout=sys.stderr)
        linearizer.process_part(part)
        return " ".join(linearizer.output_tokens)

    @staticmethod
    def _get_data(data: Path | str) -> str:
        """Read data from path or return string directly.

        Args:
            data (Path | str): File path or raw data string

        Returns:
            str: Content of file or original string
        """
        if not isinstance(data, Path):
            return data
        with open(data, "r") as f:
            return f.read()

    @staticmethod
    def lmx_to_musicxml(data: Path | str, normalize: bool = False) -> str | None:
        """Convert LMX tokens to MusicXML.

        Args:
            data (Path | str): LMX file path or raw string
            normalize (bool): Apply MuseScore normalization

        Returns:
            str | None: MusicXML content or None if normalization fails
        """
        lmx_string = MusicConverter._get_data(data)
        delinearizer = Delinearizer(errout=sys.stderr)
        delinearizer.process_text(lmx_string)
        score_etree = part_to_score(delinearizer.part_element)
        musicxml = str(
            ET.tostring(score_etree.getroot(), encoding="utf-8", xml_declaration=True),
            "utf-8",
        )
        if normalize:
            return MusicConverter.normalize_musicxml(musicxml)
        return musicxml

    @staticmethod
    def save_musicxml_to_file(
        data: str, filename: str | None = None, suffix: str = ".musicxml"
    ) -> Path:
        """Save MusicXML data to file with timestamp.

        Args:
            data (str): MusicXML content to save
            filename (str, optional): Base filename
            suffix (str): File extension/descriptor

        Returns:
            Path: Path to created file
        """
        if not filename:
            basename = "generated"
            time_info_suffix = (
                datetime.datetime.now().strftime("%y%m%d_%H%M%S") + suffix
            )
            filename = "_".join([basename, time_info_suffix])

        examples_dir = Path(BASE_DIR / "generated_examples")
        if not examples_dir.exists():
            examples_dir.mkdir()

        filepath = examples_dir / filename
        with open(filepath, "w") as f:
            f.write(data)
        return filepath
