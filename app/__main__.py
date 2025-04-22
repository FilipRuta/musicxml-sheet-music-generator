import argparse

from .lmx_generator import LMXGenerator
from .music_converter import MusicConverter
from .unity_connector import UnityServerSocket


def save_results(
    generated_seqs: list[str] | str,
    save_lmx: bool = True,
    save_music_xml: bool = True,
    save_normalized: bool = True,
) -> None:
    """Save generated sequences in various formats.

    Args:
        generated_seqs: Generated LMX sequence(s) to process
        save_lmx: Save raw LMX format
        save_music_xml: Convert to MusicXML and save
        save_normalized: Create MuseScore-normalized MusicXML version
    """
    if not isinstance(generated_seqs, list):
        generated_seqs = [generated_seqs]
    for i in range(len(generated_seqs)):
        if save_lmx:
            MusicConverter.save_musicxml_to_file(generated_seqs[i], suffix=".lmx")
        if save_music_xml:
            gen_seq_xml = MusicConverter.lmx_to_musicxml(generated_seqs[i])
            MusicConverter.save_musicxml_to_file(gen_seq_xml)
        if save_normalized:
            gen_seq_xml_norm = MusicConverter.lmx_to_musicxml(
                generated_seqs[i], normalize=True
            )
            if not gen_seq_xml_norm:
                continue
            MusicConverter.save_musicxml_to_file(
                gen_seq_xml_norm, suffix="_normalized.musicxml"
            )


def generate(
    n: int,
    model_name: str,
    seed: str | None,
    save_lmx: bool,
    save_music_xml: bool,
    normalize_with_musescore: bool,
) -> None:
    """Generate musical sequences using trained model.

    Args:
        n: Number of sequences to generate
        model_name: Model filename in trained_models directory
        seed: Optional starting sequence
        save_lmx: Preserve LMX format output
        save_music_xml: Create MusicXML versions
        normalize_with_musescore: Apply MuseScore normalization
    """
    if not save_lmx and not save_music_xml and not normalize_with_musescore:
        raise Exception("No save option specified")

    generator = LMXGenerator(load_pretrained_model=True, model_name=model_name)
    generated_sequences = generator.generate(seed, num_samples=n)
    save_results(
        generated_sequences, save_lmx, save_music_xml, normalize_with_musescore
    )
    print("Generation complete.")


def train(dataset_filename: str, checkpoint_path: str | None) -> None:
    """Train model_name on specified dataset.

    Args:
        dataset_filename: Path to preprocessed training data
        checkpoint_path: Optional checkpoint absolute path to resume training from
    """
    if not dataset_filename:
        raise Exception("Dataset not specified")
    model = LMXGenerator(load_pretrained_model=False)
    model.train(
        dataset_filename=dataset_filename, checkpoint_path_to_load=checkpoint_path
    )
    print("Training complete.")


def serve(model_name: str, host: str, port: int) -> None:
    """Launch server for Unity communication.

    Args:
        model_name: Model filename in trained_models directory
        host: Server IP/hostname
        port: Communication port
    """
    generator = LMXGenerator(load_pretrained_model=True, model_name=model_name)
    unity_server = UnityServerSocket(generator, host, port)
    unity_server.create_server()


def main() -> None:
    """Main CLI entry point for music generation pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Music Generation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Host command
    serve_parser = subparsers.add_parser(
        "serve", help="Start hosting local server for unity communication"
    )
    serve_parser.add_argument(
        "--model",
        type=str,
        help="Name of model to load from trained_models directory",
    )
    serve_parser.add_argument(
        "--host", type=str, help="Hostname or an IP address", default="127.0.0.1"
    )
    serve_parser.add_argument("--port", type=int, help="Port number", default=25001)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate music examples")
    generate_parser.add_argument(
        "-n", type=int, default=1, help="Number of examples to generate"
    )
    generate_parser.add_argument(
        "--model",
        type=str,
        help="Name of model to load from trained_models directory",
    )
    generate_parser.add_argument(
        "--save-lmx", action="store_true", help="Save LMX output to file"
    )
    generate_parser.add_argument(
        "--save-musicxml", action="store_true", help="Save MusicXML output to file"
    )
    generate_parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Normalize output using MuseScore and Save MusicXML output to file",
    )
    generate_parser.add_argument(
        "--seed", type=str, help="Add LMX seed used as start for the generator"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--train-data-filename",
        type=str,
        help="Filename of lmx data in converted_dataset dir in pickle format",
    )
    train_parser.add_argument("--checkpoint", type=str, help="Absolute path to checkpoint file")
    args = parser.parse_args()

    if args.command == "serve":
        serve(args.model, args.host, args.port)
    elif args.command == "generate":
        generate(
            args.n,
            args.model,
            args.seed,
            args.save_lmx,
            args.save_musicxml,
            args.save_normalized,
        )
    elif args.command == "train":
        train(args.train_data_filename, args.checkpoint)


if __name__ == "__main__":
    main()
