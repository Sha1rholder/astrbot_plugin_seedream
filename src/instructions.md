# Seedream image generation instructions

Generate and deliver one lossless WebP image with Seedream 5.5 using `seedream_image_generate`.

- `prompt`: Describe the subject, composition, style and constraints. Specify aspect ratio and intended use here; resolution defaults to 2K. For edits, describe what should change and what should remain unchanged, identifying references in selection order. Must be nonblank, up to 32,000 characters.
- `description`: A short caption automatically sent with the image. Does not affect generation. Must be nonblank, up to 1,000 characters.
- (Optional) `reference_images`: An ordered list of up to 10 distinct attachment indices, absolute local image paths on the AstrBot host, or publicly accessible HTTPS image URLs. All three may be mixed. Integer indices start at 0: current message images first, followed by quoted images. Available indices are listed below. Omit for text-only generation.
- (Optional) `background`: `auto`, `transparent` or `opaque`. Defaults to `auto` when omitted or null. Use `transparent` for an alpha channel.

In most cases, omit optional arguments unless needed. Each call generates one image.

```json
{"prompt":"A watercolor cat by a window, landscape composition, 16:9 aspect ratio","description":"Watercolor cat by a window"}
```

References must contain valid JPEG, PNG or WebP data, at most 30 MB (30,000,000 bytes) per image. Both dimensions must exceed 14 pixels, total pixels must not exceed 36 million, and aspect ratio must be between 1:16 and 16:1. Local paths must be readable by AstrBot and end in `.jpg`, `.jpeg`, `.png` or `.webp` (case insensitive). HTTPS URLs need no file extension; preserve all signed query parameters.

For edits, select references in the order you describe them. Attachment indices refer only to the listed current and quoted attachments, not earlier conversation history.

```json
{"prompt":"Remove the background from image 1, preserve the cat and its fine fur edges","description":"Cat with a transparent background","reference_images":[0],"background":"transparent"}
```

To find previously generated images, search the archive directory shown below with `rg -n -i -F -g request.yaml -- 'keyword' 'ARCHIVE_DIR'`. Each successful generation has a numbered directory containing `request.yaml`, `response.yaml` and `0.webp`.

For a follow-up edit, pass the existing `0.webp` absolute path or the HTTPS URL in `response.yaml` at `data[0].url`. Generated URLs are valid for 24 hours after generation; after expiry, use the local WebP while it remains in the archive. Use the file or URL belonging to the intended version. Replace the example path with an actual existing file:

```json
{"prompt":"Give the cat in image 1 the color palette of image 2, preserving its pose","description":"Cat with the reference color palette","reference_images":["/AstrBot/data/plugin_data/astrbot_plugin_seedream/images/42/0.webp",0]}
```
