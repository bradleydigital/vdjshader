import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


SHADERTOY_BASE = "https://www.shadertoy.com"

WARNING_RULES_FILE = "shader_warnings.json"
WARNING_SCOPE_LABELS = {
    "macos_metal": "macOS/Metal-only",
    "generic": "General",
}


@dataclass
class TextureAsset:
    original_path: str          # e.g. /media/a/<hash>.jpg
    local_name: str             # e.g. 0_<hash>.jpg
    content: bytes
    error_message: Optional[str] = None  # Store error message if download failed


@dataclass
class WarningRule:
    id: str
    title: str
    kind: str  # "source" | "error"
    pattern: str
    message: str
    scope: str = "generic"


@dataclass
class WarningMatch:
    rule_id: str
    title: str
    message: str
    scope: str
    where: str  # e.g. "source:pass=1(Image)" or "error:ocr"


def load_warning_rules() -> List[WarningRule]:
    """
    Loads warning rules from shader_warnings.json.
    Returns [] if missing or invalid.
    """
    try:
        with open(WARNING_RULES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rules: List[WarningRule] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            rules.append(
                WarningRule(
                    id=str(item.get("id", "")).strip(),
                    title=str(item.get("title", "")).strip(),
                    kind=str(item.get("kind", "")).strip(),
                    pattern=str(item.get("pattern", "")).strip(),
                    message=str(item.get("message", "")).strip(),
                    scope=str(item.get("scope", "generic")).strip() or "generic",
                )
            )
        # filter invalid
        rules = [r for r in rules if r.id and r.title and r.kind in ("source", "error") and r.pattern]
        return rules
    except Exception:
        return []


def extract_renderpass_codes(shadertoy_obj: Any) -> List[Dict[str, Any]]:
    """
    Returns list of {index, type, name, code}.
    """
    root = _ensure_shader_root(shadertoy_obj)
    renderpasses = list(root.get("renderpass", []))
    out: List[Dict[str, Any]] = []
    for idx, rp in enumerate(renderpasses, start=1):
        if not isinstance(rp, dict):
            continue
        out.append(
            {
                "index": idx,
                "type": rp.get("type", "") or "",
                "name": rp.get("name", "") or "",
                "code": rp.get("code", "") or "",
            }
        )
    return out


def match_rules(text: str, rules: List[WarningRule], where: str) -> List[WarningMatch]:
    matches: List[WarningMatch] = []
    if not text:
        return matches
    for rule in rules:
        try:
            if re.search(rule.pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                matches.append(
                    WarningMatch(
                        rule_id=rule.id,
                        title=rule.title,
                        message=rule.message,
                        scope=rule.scope or "generic",
                        where=where,
                    )
                )
        except re.error:
            continue
    return matches


def _ensure_shader_root(obj: Any) -> Dict[str, Any]:
    """
    Accepts Shadertoy export forms:
    - {"Shader": {...}}
    - [{...}]  (list with one shader object like user pasted)
    - {...}    (direct shader object containing "info" and "renderpass")
    Returns the dict containing keys: "info", "renderpass"
    """
    if isinstance(obj, dict) and "Shader" in obj and isinstance(obj["Shader"], dict):
        return obj["Shader"]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "renderpass" in obj[0]:
        return obj[0]
    if isinstance(obj, dict) and "renderpass" in obj:
        return obj
    raise ValueError("Could not recognise Shadertoy JSON structure. Paste the export containing info + renderpass.")


def _buffer_index_from_filepath(fp: str) -> Optional[int]:
    """
    /media/previz/buffer00.png -> 0
    /media/previz/buffer01.png -> 1
    """
    m = re.search(r"buffer(\d{2})\.png$", fp or "")
    if not m:
        return None
    return int(m.group(1))


def _buffer_index_from_name(name: str) -> Optional[int]:
    """
    Buffer A/B/C/D -> 0/1/2/3
    """
    if not name:
        return None
    m = re.search(r"Buffer\s*([ABCD])", name, re.IGNORECASE)
    if not m:
        return None
    letter = m.group(1).upper()
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(letter)


def _xml_type_for_buffer_index(buf_idx: int) -> int:
    """
    VirtualDJ mapping we inferred from your working packages:
    buffer00 -> type=1
    buffer01 -> type=2
    buffer02 -> type=3
    buffer03 -> type=4
    """
    return buf_idx + 1


def _download_asset(path: str, timeout: int = 20) -> bytes:
    """
    Downloads /media/... assets from shadertoy.com
    """
    url = SHADERTOY_BASE + path
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.shadertoy.com/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        if len(r.content) == 0:
            raise Exception(f"Downloaded empty file from {path}")
        return r.content
    except requests.exceptions.HTTPError as e:
        status_code = r.status_code if hasattr(r, 'status_code') else 'unknown'
        status_text = r.reason if hasattr(r, 'reason') else str(e)
        raise Exception(f"HTTP {status_code} ({status_text}) error downloading {path}. URL: {url}") from e
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error downloading {path}: {e}. URL: {url}") from e


def _make_local_texture_name(path: str) -> str:
    """
    /media/a/<hash>.jpg -> 0_<hash>.jpg
    /media/??/<hash>.png -> 0_<hash>.png
    """
    base = os.path.basename(path)
    return f"0_{base}"


def _fix_metal_keyword_conflicts(code: str) -> str:
    """
    Automatically renames variables/parameters that conflict with Metal keywords.
    Metal keywords: or, and, not, xor, new
    
    Only renames if we find actual declarations. Uses word boundaries to avoid
    replacing operators (||, &&) or keywords in comments/strings.
    """
    metal_keywords = ["or", "and", "not", "xor", "new"]
    rename_map = {}
    
    # Find all variable/parameter declarations that use Metal keywords
    # Pattern: type keyword (e.g., "float or", "int and", "vec2 not")
    # Also match in function parameters: (float or, ...)
    for keyword in metal_keywords:
        # Match type declarations: float/int/vec*/mat*/bool followed by the keyword
        # This matches: "float or", "vec2 new", etc. in declarations
        pattern = r'\b(float|int|vec[234]|mat[234]|bool|sampler2D)\s+(' + re.escape(keyword) + r')\b'
        matches = re.finditer(pattern, code)
        
        for match in matches:
            var_name = match.group(2)
            if var_name not in rename_map:
                # Create a safe replacement name (e.g., "or" -> "or_metal")
                new_name = f"{var_name}_metal"
                rename_map[var_name] = new_name
    
    # Only perform replacements if we found actual declarations
    # This ensures we don't touch shaders that don't have keyword conflicts
    if not rename_map:
        return code
    
    # Perform replacements throughout the code
    # Use word boundaries to avoid partial matches and operator conflicts
    # Word boundary \b ensures we don't match:
    # - Operators: || (logical OR), && (logical AND)
    # - Part of other words: "newValue" won't match "new"
    for old_name, new_name in rename_map.items():
        # Replace all occurrences of the variable name, ensuring word boundaries
        code = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, code)
    
    return code


def _convert_sampler(st_sampler: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """
    Normalise Shadertoy sampler to something VirtualDJ accepts.
    VirtualDJ seems tolerant; keep fields if present.
    """
    sampler = dict(st_sampler or {})
    # VirtualDJ examples often use vflip true for buffers, false for music.
    # We'll keep whatever is provided, defaulting sensibly.
    if "vflip" not in sampler:
        sampler["vflip"] = "true" if kind in ("buffer", "texture") else "false"
    return sampler


def build_vdj_from_shadertoy(
    shadertoy_obj: Any,
    vdj_id_override: Optional[str] = None,
    embed_textures: bool = True,
    manual_textures: Optional[Dict[str, bytes]] = None,
    compatibility_mode: bool = False,
) -> Tuple[Dict[str, Any], str, List[TextureAsset]]:
    """
    Returns:
    - vdj_shader_json_root (dict with key "Shader")
    - shader_xml string
    - list of TextureAsset to embed
    """
    root = _ensure_shader_root(shadertoy_obj)
    info = dict(root.get("info", {}))
    renderpasses = list(root.get("renderpass", []))

    vdj_id = vdj_id_override or info.get("id") or "vdjshader"
    vdj_name = info.get("name") or vdj_id

    # --- Collect textures to embed and build XML src names ---
    textures: List[TextureAsset] = []
    texture_src_map: Dict[str, str] = {}  # original /media/... -> local file name

    if embed_textures:
        for rp in renderpasses:
            for inp in rp.get("inputs", []) or []:
                if inp.get("type") == "texture":
                    fp = inp.get("filepath") or inp.get("src")  # shadertoy uses filepath
                    if fp and fp not in texture_src_map:
                        local_name = _make_local_texture_name(fp)
                        content = b""
                        
                        # Check if manually uploaded texture exists for this path
                        if manual_textures and fp in manual_textures:
                            content = manual_textures[fp]
                            textures.append(TextureAsset(fp, local_name, content))
                            texture_src_map[fp] = local_name
                        else:
                            # Try to download
                            try:
                                content = _download_asset(fp)
                                textures.append(TextureAsset(fp, local_name, content))
                                texture_src_map[fp] = local_name
                            except Exception:
                                # If download fails, add texture with empty content
                                textures.append(TextureAsset(fp, local_name, b""))
                                texture_src_map[fp] = local_name

    # --- Build VDJ shader.json ---
    # VirtualDJ wrapper expects {"Shader": {...}} like your working files
    vdj_info = dict(info)
    vdj_info["id"] = vdj_id
    vdj_info["name"] = vdj_name

    vdj_renderpass: List[Dict[str, Any]] = []

    # Helper: assign numeric IDs to buffers (matching VDJ conventions you saw)
    # buffer00 -> 257, buffer01 -> 258, buffer02 -> 259, buffer03 -> 260
    def buffer_numeric_id(buf_idx: int) -> int:
        return 257 + buf_idx

    for rp in renderpasses:
        rp_type = rp.get("type", "")
        rp_name = rp.get("name", "") or ""
        rp_code = rp.get("code", "")
        
        # Fix shader version for VirtualDJ compatibility
        # VirtualDJ doesn't support #version 310 es, remove version directives entirely
        # VirtualDJ will add its own compatible version directive
        if rp_code:
            # Remove all version directives - VirtualDJ will add its own
            rp_code = re.sub(r'#version\s+\d+\s*(es)?\s*\n?', '', rp_code, flags=re.IGNORECASE | re.MULTILINE)
            
            # Fix Metal keyword conflicts by renaming variables/parameters
            rp_code = _fix_metal_keyword_conflicts(rp_code)

            # Compatibility mode: avoid array uniforms that can break VirtualDJ/Metal compilation
            # - iChannelTime[n] -> iTime
            # - iChannelResolution[n] -> iResolution
            if compatibility_mode:
                rp_code = re.sub(r"\biChannelTime\s*\[\s*\d+\s*\]", "iTime", rp_code)
                rp_code = re.sub(r"\biChannelResolution\s*\[\s*\d+\s*\]", "iResolution", rp_code)
                # Some shaders use iFrameRate (not always provided by VirtualDJ). Replace with a sane default.
                rp_code = re.sub(r"\biFrameRate\b", "60.0", rp_code)

        # Inputs conversion
        vdj_inputs: List[Dict[str, Any]] = []
        for inp in rp.get("inputs", []) or []:
            inp_type = inp.get("type")
            channel = int(inp.get("channel", 0))

            if inp_type in ("music", "musicstream", "mic"):
                # Shadertoy uses filepath; VDJ uses src
                fp = inp.get("filepath") or ""
                # Normalize preset paths (like /presets/mic.png) to default audio path
                # VirtualDJ doesn't recognize Shadertoy preset paths
                if fp.startswith("/presets/"):
                    fp = "/media/a/a6a1cf7a09adfed8c362492c88c30d74fb3d2f4f7ba180ba34b98556660fada1.mp3"
                vdj_inputs.append({
                    "id": 19,
                    "src": fp.replace("\\/", "/") if fp else "/media/a/a6a1cf7a09adfed8c362492c88c30d74fb3d2f4f7ba180ba34b98556660fada1.mp3",
                    "ctype": "music",
                    "channel": channel,
                    "sampler": _convert_sampler(inp.get("sampler", {}), "music"),
                    "published": int(inp.get("published", 1)),
                })
            elif inp_type == "buffer":
                fp = inp.get("filepath") or inp.get("previewfilepath") or ""
                idx = _buffer_index_from_filepath(fp)
                if idx is None:
                    # fallback: guess from renderpass name if it refers to Buffer A etc
                    idx = _buffer_index_from_name(rp_name) or 0
                vdj_inputs.append({
                    "id": buffer_numeric_id(idx),
                    "src": f"/media/previz/buffer{idx:02d}.png",
                    "ctype": "buffer",
                    "channel": channel,
                    "sampler": _convert_sampler(inp.get("sampler", {}), "buffer"),
                    "published": int(inp.get("published", 1)),
                })
            elif inp_type == "texture":
                fp = inp.get("filepath") or ""
                vdj_inputs.append({
                    "id": 46,
                    "src": fp,
                    "ctype": "texture",
                    "channel": channel,
                    "sampler": _convert_sampler(inp.get("sampler", {}), "texture"),
                    "published": int(inp.get("published", 1)),
                })
            else:
                # Unknown channel types: keep as-is but mark ctype so you can spot it
                # VirtualDJ may reject these; you'll see in compile errors.
                fp = inp.get("filepath") or ""
                vdj_inputs.append({
                    "id": 999,
                    "src": fp,
                    "ctype": inp_type or "unknown",
                    "channel": channel,
                    "sampler": _convert_sampler(inp.get("sampler", {}), "unknown"),
                    "published": int(inp.get("published", 1)),
                })

        # Outputs conversion
        vdj_outputs: List[Dict[str, Any]] = []
        if rp_type == "buffer":
            # Decide which buffer this pass writes to
            # Prefer name Buffer A/B/C/D else infer from its output filepath if present
            buf_idx = _buffer_index_from_name(rp_name)
            if buf_idx is None:
                outs = rp.get("outputs", []) or []
                # Shadertoy output objects often include only id/channel; no filepath.
                # So we fallback to sequential order of buffer passes:
                # count how many buffer passes before this one.
                buffer_passes_before = sum(1 for prev in renderpasses[:renderpasses.index(rp)] if prev.get("type") == "buffer")
                buf_idx = min(buffer_passes_before, 3)

            vdj_outputs = [{"id": buffer_numeric_id(buf_idx), "channel": 0}]

        # VDJ expects outputs [] for image
        vdj_renderpass.append({
            "inputs": vdj_inputs,
            "outputs": vdj_outputs if rp_type == "buffer" else [],
            "code": rp_code,
            "name": rp_name,
            "description": rp.get("description", ""),
            "type": rp_type,
        })

    vdj_shader_json = {"Shader": {"ver": root.get("ver", "0.1"), "info": vdj_info, "renderpass": vdj_renderpass}}

    # --- Build shader.xml ---
    # Pass numbering starts at 1
    xml_lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<shaderinfo svn="9004" id="{vdj_id}" name="{_xml_escape(vdj_name)}" nbpasses="{len(vdj_renderpass)}">'
    ]

    for i, rp in enumerate(vdj_renderpass, start=1):
        rp_type = rp["type"]
        rp_name = rp.get("name", "")

        if rp_type == "buffer":
            # Give buffer passes an id attribute (1..4)
            buf_idx = _buffer_index_from_name(rp_name)
            if buf_idx is None:
                # infer by counting buffer passes up to this pass
                buf_idx = sum(1 for p in vdj_renderpass[:i] if p["type"] == "buffer") - 1
                buf_idx = max(0, min(buf_idx, 3))
            xml_lines.append(f'\t<renderpass pass="{i}" type="buffer" id="{buf_idx+1}" codeflag="0">')
        else:
            xml_lines.append(f'\t<renderpass pass="{i}" type="image" codeflag="0">')

        # Inputs
        for inp in (renderpasses[i-1].get("inputs", []) or []):
            ch = int(inp.get("channel", 0))
            t = inp.get("type")

            if t in ("music", "musicstream", "mic"):
                xml_lines.append(f'\t\t<input channel="{ch}" type="7" flag="0" />')
            elif t == "buffer":
                fp = inp.get("filepath") or inp.get("previewfilepath") or ""
                bidx = _buffer_index_from_filepath(fp)
                if bidx is None:
                    bidx = 0
                xml_lines.append(f'\t\t<input channel="{ch}" type="{_xml_type_for_buffer_index(bidx)}" flag="0" />')
            elif t == "texture":
                fp = inp.get("filepath") or ""
                local_name = texture_src_map.get(fp) or _make_local_texture_name(fp)
                # VirtualDJ pattern from your working shader packages: type=5 flag=3 src=<file inside zip>
                xml_lines.append(f'\t\t<input channel="{ch}" type="5" flag="3" src="{_xml_escape(local_name)}" />')
            else:
                # Unknown: keep as texture-like to avoid missing input line, but warn in UI.
                xml_lines.append(f'\t\t<input channel="{ch}" type="5" flag="0" />')

        xml_lines.append('\t</renderpass>')

    xml_lines.append('</shaderinfo>')
    shader_xml = "\n".join(xml_lines)

    return vdj_shader_json, shader_xml, textures


def _xml_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def package_vdjshader_bytes(
    vdj_shader_json: Dict[str, Any],
    shader_xml: str,
    textures: List[TextureAsset],
    filename_base: str,
) -> bytes:
    """
    Builds a .vdjshader (zip) in-memory.
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("shader.json", json.dumps(vdj_shader_json, ensure_ascii=False, separators=(",", ":")))
        z.writestr("shader.xml", shader_xml)

        # embed textures
        for tex in textures:
            if tex.content:
                z.writestr(tex.local_name, tex.content)

    return mem.getvalue()


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Shadertoy → VirtualDJ (.vdjshader)", layout="wide")
st.title("Shadertoy → VirtualDJ .vdjshader exporter")

st.write(
    "Paste the Shadertoy export JSON you copied from DevTools (Network → the API response containing `info` + `renderpass`). "
    "This app will generate a VirtualDJ `.vdjshader` file (shader.json + shader.xml + embedded textures)."
)

def _on_raw_change() -> None:
    # Triggered when Streamlit registers a change for the text area.
    # Note: Streamlit's built-in textarea updates on blur / Ctrl+Enter.
    st.session_state["_raw_last_edited"] = True


raw = st.text_area(
    "Paste Shadertoy JSON here",
    height=320,
    placeholder='Paste the JSON response (can be a list or {"Shader":...})',
    key="raw_json",
    on_change=_on_raw_change,
)
st.caption("Tip: press Ctrl+Enter to apply changes without clicking outside the text area.")

embed_textures = st.checkbox("Download + embed textures from Shadertoy", value=True)

# Auto-fixes for known VirtualDJ/Metal incompatibilities (optional)
compatibility_mode = st.checkbox(
    "Compatibility mode (auto-fix common Metal issues)",
    value=False,
    help="When enabled: rewrites iChannelTime[n] -> iTime and iChannelResolution[n] -> iResolution to avoid certain VirtualDJ/Metal compilation failures.",
)

# ------------------ Live compatibility warnings ------------------
if "warning_rules" not in st.session_state:
    st.session_state.warning_rules = load_warning_rules()

warning_rules: List[WarningRule] = st.session_state.warning_rules
warning_matches: List[WarningMatch] = []

parsed: Optional[Any] = None
if raw:
    try:
        parsed = json.loads(raw)
        for rp in extract_renderpass_codes(parsed):
            code = rp.get("code", "") or ""
            if not code:
                continue
            pass_label = f'pass={rp.get("index")}({(rp.get("name") or rp.get("type") or "").strip()})'
            warning_matches.extend(match_rules(code, [r for r in warning_rules if r.kind == "source"], where=f"source:{pass_label}"))
    except Exception:
        parsed = None

st.divider()
st.subheader("Compatibility warnings (informational)")
st.caption("These warnings are heuristics based on patterns from shaders that failed to compile in VirtualDJ.")

if warning_matches:
    # de-dup by (rule_id, where)
    seen = set()
    deduped: List[WarningMatch] = []
    for m in warning_matches:
        key = (m.rule_id, m.where)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    by_scope: Dict[str, List[WarningMatch]] = {}
    for m in deduped:
        by_scope.setdefault(m.scope or "generic", []).append(m)

    for scope_key in ["macos_metal", "generic"]:
        if scope_key not in by_scope:
            continue
        st.warning(f'{WARNING_SCOPE_LABELS.get(scope_key, scope_key)} warnings')
        for m in by_scope[scope_key]:
            where_suffix = f" ({m.where})" if m.where else ""
            st.markdown(f"- `{m.title}`{where_suffix}\n\n  {m.message}")
else:
    st.info("No known warning patterns detected yet.")

# Extract texture info from JSON to show download buttons
texture_paths = []
if raw and parsed is not None:
    try:
        root = _ensure_shader_root(parsed)
        renderpasses = root.get("renderpass", [])
        seen_paths = set()
        for rp in renderpasses:
            for inp in rp.get("inputs", []) or []:
                if inp.get("type") == "texture":
                    fp = inp.get("filepath") or inp.get("src")
                    if fp and fp not in seen_paths:
                        texture_paths.append(fp)
                        seen_paths.add(fp)
    except:
        pass

# Show texture download buttons
if texture_paths:
    st.divider()
    st.subheader("Textures")
    st.caption("Download textures manually if automatic download fails, then upload them below.")
    for tex_path in texture_paths:
        tex_url = SHADERTOY_BASE + tex_path
        tex_filename = os.path.basename(tex_path)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(tex_filename)
        with col2:
            try:
                st.link_button("Download", tex_url)
            except:
                st.markdown(f"[Download]({tex_url})")
    
    st.divider()
    st.subheader("Upload Textures")
    uploaded_textures = st.file_uploader(
        "Upload texture files",
        type=["jpg", "jpeg", "png", "gif", "bmp"],
        accept_multiple_files=True,
        help="Upload texture files you downloaded above. They will be matched by filename."
    )
else:
    uploaded_textures = None

if st.button("Generate .vdjshader", type="primary"):
    try:
        parsed = json.loads(raw)
        
        # Get shadertoy ID for filename
        root = _ensure_shader_root(parsed)
        shadertoy_id = root.get("info", {}).get("id") or "vdjshader"
        
        # Process manually uploaded textures
        manual_textures_dict = {}
        if uploaded_textures:
            # Extract texture paths from JSON to match against
            renderpasses = root.get("renderpass", [])
            texture_paths_map = {}
            for rp in renderpasses:
                for inp in rp.get("inputs", []) or []:
                    if inp.get("type") == "texture":
                        fp = inp.get("filepath") or inp.get("src")
                        if fp:
                            # Store by basename for matching
                            basename = os.path.basename(fp)
                            texture_paths_map[basename] = fp
            
            # Match uploaded files to texture paths
            for uploaded_file in uploaded_textures:
                uploaded_basename = uploaded_file.name
                # Try to find matching texture path
                for tex_basename, tex_path in texture_paths_map.items():
                    if uploaded_basename == tex_basename or uploaded_basename.endswith(tex_basename):
                        manual_textures_dict[tex_path] = uploaded_file.read()
                        break
        
        vdj_json, vdj_xml, assets = build_vdj_from_shadertoy(
            parsed,
            vdj_id_override=None,
            embed_textures=embed_textures,
            manual_textures=manual_textures_dict if manual_textures_dict else None,
            compatibility_mode=compatibility_mode,
        )

        vdjshader_bytes = package_vdjshader_bytes(vdj_json, vdj_xml, assets, shadertoy_id)

        st.success("Generated successfully.")
        st.download_button(
            label="Download .vdjshader",
            data=vdjshader_bytes,
            file_name=f"{shadertoy_id}.vdjshader",
            mime="application/zip",
        )

        with st.expander("Preview generated shader.xml"):
            st.code(vdj_xml, language="xml")

        with st.expander("Preview generated shader.json (minified)"):
            st.code(json.dumps(vdj_json, ensure_ascii=False, separators=(",", ":"))[:5000] + "\n...", language="json")

    except Exception as e:
        st.error(f"Failed: {e}")
