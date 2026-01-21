# Shadertoy to VirtualDJ Shader Converter

A Python app that converts Shadertoy JSON exports to VirtualDJ `.vdjshader` files.

## Features

- Converts Shadertoy shader JSON (from DevTools) to VirtualDJ format
- Automatically downloads and embeds textures from Shadertoy
- Generates both `shader.json` and `shader.xml` files
- Packages everything into a `.vdjshader` zip file ready for VirtualDJ

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Optional: OCR for error screenshots

If you want to upload VirtualDJ error screenshots and have the app extract the text automatically, install **Tesseract OCR** on your system.

- macOS (Homebrew):
```bash
brew install tesseract
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. In Shadertoy:
   - Open DevTools (F12)
   - Go to Network tab
   - Load a shader page
   - Find the API response containing `info` and `renderpass` fields
   - Copy the JSON response

4. In the app:
   - Paste the JSON into the text area
   - Optionally set a VirtualDJ ID and output filename
   - Choose whether to embed textures
   - Click "Generate .vdjshader"
   - Download the generated file

## How It Works

The app:
- Parses Shadertoy JSON (supports multiple formats: `{"Shader": {...}}`, `[{...}]`, or direct shader objects)
- Converts render passes, inputs, and outputs to VirtualDJ format
- Maps buffer indices (Buffer A/B/C/D → buffer00/buffer01/buffer02/buffer03)
- Downloads texture assets from Shadertoy if enabled
- Generates `shader.xml` with proper VirtualDJ XML structure
- Packages everything into a `.vdjshader` zip file

## Notes

- Music inputs are mapped to XML type=7 (VirtualDJ standard)
- Buffers map to: buffer00→type=1, buffer01→type=2, buffer02→type=3, buffer03→type=4
- Textures are embedded in the zip file and referenced in the XML
- If texture downloads fail, the shader may still work but textures might be missing
