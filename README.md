## MusicXML sheet music generator

This MusicXML generator was developed as a component of a game designed to help users learn to read music notation. The game
can be accessed [here](https://github.com/FilipRuta/sight-readia)

### Installation (For windows)

1. Download the generator from releases.
2. Install the *MuseScore* application (used for music normalization). You can download it from [here](https://musescore.org/en/download). The version used during release was *MuseScore 4.4.2*.
3. Locate the installed *MuseScore* application on your PC. By default, it is installed at:
   ```
   C:\Program Files\MuseScore 4\bin\MuseScore4.exe
   ```
4. In the generator directory, open `config.yaml` in any text editor and set the correct path to your *MuseScore* application.
5. Make sure you are connected to the internet. To install all requirements, run:
   ```
   install_requirements.bat
   ```
   Wait for the installation to complete.
6. To verify that the setup is correct, run:
   ```
   create_server.bat
   ```
   After a few seconds, you should see the message:
   ```
   MuseScore installation path set successfully
   ```
7. The generator is now functional (CPU-only mode). For GPU acceleration, follow the steps below.

### Optional: Enable GPU Support (NVIDIA)

To use your Nvidia GPU for faster generation:

1. Open the command line by searching for `cmd` in the Windows search bar.
2. Enter the command:
   ```
   nvidia-smi --version
   ```
   This will show your CUDA version (e.g., 12.6). If the command is not found, your GPU might be unsupported, or drivers may be missing.
3. Ensure your CUDA version is one of the supported versions (11.8, 12.4, or 12.6).
4. Run the following file:
   ```
   install_cuda.bat
   ```
   A console window will appear. Enter your CUDA version when prompted and press Enter. The required packages will be downloaded and installed.
5. After the installation is complete, run:
   ```
   create_server.bat
   ```
   You should now see the message:
   ```
   Model is running on cuda
   ```

---

### How to Use

#### Running a Server

To start the generator server, run:
```
create_server.bat
```
If you see the message:
```
Listening on host:port
```
it means the server is running and ready to communicate with the game.

#### Console Mode

You can also start an interactive console by running:
```
python_console.bat
```
This opens a Python interpreter environment bundled with the generator.

To run the server manually, type:
```
python -m app serve
```

Optional parameters:
- `--model model_name` – use a model from the `trained_models` directory
- `--host host_name` – specify a host or IP address
- `--port port` – specify a port
- `-h` or `--help` – show available options

#### Generating Songs

To generate songs directly and save them to files, run:
```
python -m app generate --save-musicxml
```

The results will be saved in 

``
app/generated_examples/
``

Optional parameters:
- `-n` – number of songs to generate
- `--model model_name` – use a specific model from the `trained_models` directory
- `--save-lmx` – save the output in `.lmx` format
- `--save-musicxml` – save the output in uncompressed `.musicxml` format
- `--save-normalized` – normalize and save using MuseScore
- `--seed "lmx token sequence"` – provide an initial seed (e.g., `"measure key:fifths:2 clef:G2 staff:1 clef:F4 staff:2"`)
- `-h` or `--help` – show all options


## References

This generator is built on the foundation of a successful tool for linearizing and delinearizing the MusicXML format, 
which has significantly influenced and enabled our work. 

Jiří Mayer, Milan Straka, Jan Hajič jr., Pavel Pecina. Practical End-to-End Optical Music Recognition for Pianoform Music.
*18th International Conference on Document Analysis and Recognition, ICDAR 2024.* 
Athens, Greece, August 30 - September 4, pp. 55-73, 2024.<br>
**DOI:** https://doi.org/10.1007/978-3-031-70552-6_4<br>
**GitHub:** https://github.com/ufal/olimpic-icdar24